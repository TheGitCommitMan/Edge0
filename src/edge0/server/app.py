"""HTTP app layer (Flask optional; stdlib fallback).

Preference order: if ``flask`` is importable it is used (the full
OpenAI-compatible surface with SSE streaming); otherwise a minimal
stdlib ``http.server`` handler serves the same JSON contract and SSE
streaming.  Both share ``build_app_handlers`` so the route
semantics stay identical.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from edge0.server.chat import (ChatMessage, ChatRequest, QueueServer,
                               decode_tokens, parse_chat_request, sse_format)

try:  # pragma: no cover - environment dependent
    from flask import Flask, Response, jsonify, request  # type: ignore

    _HAS_FLASK = True
except ImportError:  # pragma: no cover
    _HAS_FLASK = False


def _error(status: int, msg: str) -> tuple:
    if _HAS_FLASK:
        return jsonify({"error": {"message": msg, "type": "invalid_request"}}), status
    return (json.dumps({"error": {"message": msg}}), status)


def _split_think(text: str, think: bool):
    """With thinking enabled the model
    streams the reasoning block first and closes it with ``</think>``
    (token 156904); everything before is ``reasoning_content``, after
    is ``content``.  A generation cut off inside the reasoning block
    surfaces what it has so the UI isn't a blank wait."""
    if think:
        if "</think>" in text:
            r, c = text.split("</think>", 1)
            return r.strip(), c.strip()
        return text.strip(), ""
    return "", text


def _chat_once(server: QueueServer, payload: dict):
    req = parse_chat_request(payload)
    tokens, meta = server.chat(req)
    text = decode_tokens(server.engine, tokens)
    think = bool(req.enable_thinking if req.enable_thinking is not None
                 else getattr(server.engine, "think", False))
    reasoning, content = _split_think(text, think)
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": server.model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "reasoning_content": reasoning,
            },
            "finish_reason": "stop",
        }],
        "usage": meta["usage"],
    }


def _chat_stream(server: QueueServer, payload: dict):
    req = parse_chat_request(payload)
    events = queue.Queue()
    finished = object()
    request_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())

    def on_token(tid: int):
        text = decode_tokens(server.engine, [tid])
        events.put(sse_format({
            "id": request_id, "object": "chat.completion.chunk",
            "created": created, "model": server.model_name,
            "choices": [{"index": 0,
                         "delta": {"content": text},
                         "finish_reason": None}],
        }).encode("utf-8"))

    def produce():
        try:
            _, meta = server.chat(req, on_token=on_token)
            events.put(sse_format({
                "id": request_id, "object": "chat.completion.chunk",
                "created": created, "model": server.model_name,
                "choices": [{"index": 0, "delta": {},
                             "finish_reason": "stop"}],
                "usage": meta["usage"],
            }).encode("utf-8"))
            events.put(b"data: [DONE]\n\n")
        except Exception as exc:  # pragma: no cover - transport-dependent
            events.put(sse_format({
                "error": {"message": str(exc), "type": "server_error"},
            }).encode("utf-8"))
        finally:
            events.put(finished)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        event = events.get()
        if event is finished:
            return
        yield event


def _parse_anthropic_system(sys_val):
    if not sys_val:
        return ""
    if isinstance(sys_val, str):
        return sys_val
    if isinstance(sys_val, list):
        return "\n".join(
            p.get("text", "") for p in sys_val if isinstance(p, dict)
        )
    return str(sys_val)


def _parse_anthropic_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif p.get("type") == "tool_use":
                    parts.append(f"[tool_use: {p.get('name')}({json.dumps(p.get('input', {}))})]")
                elif p.get("type") == "tool_result":
                    parts.append(f"[tool_result: {p.get('content', '')}]")
        return "\n".join(parts)
    return str(content)


def _is_session_title_request(payload: dict) -> bool:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("query_source") == "generate_session_title":
        return True
    if payload.get("query_source") == "generate_session_title":
        return True
    for m in payload.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, str):
            c = content.lower()
            if "generate a title" in c or "generate a short title" in c or "session title" in c:
                return True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    p = part.get("text", "").lower()
                    if "generate a title" in p or "generate a short title" in p or "session title" in p:
                        return True
    return False


def _fast_session_title(payload: dict) -> str:
    for m in reversed(payload.get("messages", [])):
        content = m.get("content", "")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    break
        words = [w.strip("`'\".,!?:;()") for w in text.split() if w.strip("`'\".,!?:;()")]
        meaningful = [w for w in words if w.lower() not in (
            "generate", "title", "summarize", "session", "user", "human", "assistant",
            "a", "an", "the", "for", "please", "short", "very", "2-5", "words"
        )]
        if meaningful:
            return " ".join(meaningful[:4]).title()
    return "Edge0 Coding Session"


def _anthropic_to_chat_request(payload: dict) -> ChatRequest:
    msgs = []
    sys_text = _parse_anthropic_system(payload.get("system"))
    if sys_text:
        msgs.append(ChatMessage(role="system", content=sys_text))
    for m in payload.get("messages", []):
        role = str(m.get("role", "user"))
        content = _parse_anthropic_content(m.get("content", ""))
        msgs.append(ChatMessage(role=role, content=content))
    thinking_req = payload.get("thinking")
    enable_thinking = False
    if isinstance(thinking_req, dict) and thinking_req.get("type") == "enabled":
        enable_thinking = True
    return ChatRequest(
        model=str(payload.get("model", "")),
        messages=msgs,
        temperature=payload.get("temperature"),
        top_p=payload.get("top_p"),
        top_k=payload.get("top_k"),
        max_tokens=payload.get("max_tokens"),
        enable_thinking=enable_thinking,
        stream=bool(payload.get("stream", False)),
        raw=payload,
    )


def _anthropic_once(server: QueueServer, payload: dict):
    if _is_session_title_request(payload):
        title = _fast_session_title(payload)
        return {
            "id": f"msg_{int(time.time() * 1000)}",
            "type": "message",
            "role": "assistant",
            "model": server.model_name,
            "content": [{"type": "text", "text": title}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": len(title.split()),
            },
        }

    req = _anthropic_to_chat_request(payload)
    tokens, meta = server.chat(req)
    text = decode_tokens(server.engine, tokens)
    think = bool(req.enable_thinking if req.enable_thinking is not None
                 else getattr(server.engine, "think", False))
    _, content = _split_think(text, think)
    msg_id = f"msg_{int(time.time() * 1000)}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": server.model_name,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": meta["usage"]["prompt_tokens"],
            "output_tokens": meta["usage"]["completion_tokens"],
        },
    }


def _anthropic_sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _anthropic_stream(server: QueueServer, payload: dict):
    if _is_session_title_request(payload):
        title = _fast_session_title(payload)
        msg_id = f"msg_{int(time.time() * 1000)}"
        yield _anthropic_sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": server.model_name,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        })
        yield _anthropic_sse("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _anthropic_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": title},
        })
        yield _anthropic_sse("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        })
        yield _anthropic_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": len(title.split())},
        })
        yield _anthropic_sse("message_stop", {"type": "message_stop"})
        return

    req = _anthropic_to_chat_request(payload)
    events = queue.Queue()
    finished = object()
    msg_id = f"msg_{int(time.time() * 1000)}"

    def on_token(tid: int):
        text = decode_tokens(server.engine, [tid])
        if text:
            events.put(_anthropic_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }))

    def produce():
        try:
            events.put(_anthropic_sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": server.model_name,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            }))
            events.put(_anthropic_sse("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }))
            _, meta = server.chat(req, on_token=on_token)
            events.put(_anthropic_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            }))
            events.put(_anthropic_sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "output_tokens": meta["usage"]["completion_tokens"],
                },
            }))
            events.put(_anthropic_sse("message_stop", {
                "type": "message_stop",
            }))
        except Exception as exc:
            events.put(_anthropic_sse("error", {
                "type": "error",
                "error": {"type": "api_error", "message": str(exc)},
            }))
        finally:
            events.put(finished)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        event = events.get()
        if event is finished:
            return
        yield event


def build_app_handlers(server: QueueServer):
    """Return a handler dispatch dict shared by both transports."""

    def handle_chat(payload: dict):
        if payload.get("stream"):
            if _HAS_FLASK:
                return Response(
                    _chat_stream(server, payload),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return _chat_stream(server, payload)
        return _chat_once(server, payload)

    def handle_messages(payload: dict):
        if payload.get("stream"):
            if _HAS_FLASK:
                return Response(
                    _anthropic_stream(server, payload),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                    },
                )
            return _anthropic_stream(server, payload)
        return _anthropic_once(server, payload)

    handlers = {
        "GET /healthz": lambda: {"status": "ok", "model": server.model_name},
        "GET /v1/models": lambda: {
            "object": "list",
            "data": [{
                "id": server.model_name,
                "object": "model",
                "owned_by": "edge0",
            }],
        },
        "POST /v1/chat/completions": handle_chat,
        "POST /v1/messages": handle_messages,
        "POST /v1/completions": lambda p: _error(
            400, "text completions not supported; use /v1/chat/completions"),
    }
    return handlers


def create_app(server: QueueServer):
    """Flask app (raises ImportError when flask is missing)."""
    if not _HAS_FLASK:
        raise ImportError("flask is required for create_app(); "
                          "use run_stdlib instead")
    app = Flask("edge0")
    handlers = build_app_handlers(server)

    @app.get("/healthz")
    def healthz():
        return handlers["GET /healthz"]()

    @app.get("/v1/models")
    def models():
        return handlers["GET /v1/models"]()

    @app.post("/v1/chat/completions")
    def chat():
        payload = request.get_json(force=True, silent=True) or {}
        out = handlers["POST /v1/chat/completions"](payload)
        if isinstance(out, Response):
            return out
        if isinstance(out, tuple) and out and isinstance(out[0], Response):
            return out
        return jsonify(out)

    @app.post("/v1/messages")
    def messages():
        payload = request.get_json(force=True, silent=True) or {}
        out = handlers["POST /v1/messages"](payload)
        if isinstance(out, Response):
            return out
        if isinstance(out, tuple) and out and isinstance(out[0], Response):
            return out
        return jsonify(out)

    @app.post("/v1/completions")
    def completions():
        payload = request.get_json(force=True, silent=True) or {}
        out = handlers["POST /v1/completions"](payload)
        if isinstance(out, tuple):
            return out
        return jsonify(out)

    return app


class _StdlibHandler(BaseHTTPRequestHandler):
    server_q = None  # type: QueueServer
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass

    def _json(self, status: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _sse(self, events):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for body in events:
                if not body:
                    continue
                size = f"{len(body):X}".encode("ascii")
                self.wfile.write(size + b"\r\n" + body + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        handlers = build_app_handlers(self.server_q)
        if path == "/healthz":
            self._json(200, handlers["GET /healthz"]())
        elif path == "/v1/models":
            self._json(200, handlers["GET /v1/models"]())
        else:
            self._json(404, {"error": {"message": f"no route {path}"}})

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        handlers = build_app_handlers(self.server_q)
        if path not in ("/v1/chat/completions", "/v1/completions", "/v1/messages"):
            self._json(404, {"error": {"message": f"no route {path}"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if path == "/v1/messages" and payload.get("stream"):
            self._sse(_anthropic_stream(self.server_q, payload))
            return
        if path == "/v1/chat/completions" and payload.get("stream"):
            self._sse(_chat_stream(self.server_q, payload))
            return
        out = handlers[("POST " + path)](payload)
        if isinstance(out, tuple):
            body, status, headers = out
            try:
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._json(200, out)

    def log_message(self, fmt, *args):  # quiet by default
        pass


def run_stdlib(server: QueueServer, host: str, port: int):
    handler = type("Edge0Handler", (_StdlibHandler,),
                   {"server_q": server})
    httpd = ThreadingHTTPServer((host, port), handler)
    try:
        httpd.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass
    httpd.serve_forever()


def run_server(server: QueueServer, host: str = "127.0.0.1",
               port: int = 8000, use_flask: bool | None = None):
    """Run the HTTP server in the foreground (blocking)."""
    if use_flask is None:
        use_flask = _HAS_FLASK
    if use_flask:
        app = create_app(server)
        app.run(host=host, port=port, threaded=True)
    else:
        run_stdlib(server, host, port)
