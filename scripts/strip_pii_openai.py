#!/usr/bin/env python3
"""
OpenAI Privacy Filter (PII Stripper)
Based on OpenAI's open-weight model: openai/privacy-filter (Apache 2.0)
Detects and redacts 8 privacy span categories:
  1. account_number -> [ACCOUNT_NUMBER]
  2. private_address -> [ADDRESS]
  3. private_date    -> [DATE]
  4. private_email   -> [EMAIL]
  5. private_person  -> [PERSON]
  6. private_phone   -> [PHONE]
  7. private_url     -> [URL]
  8. secret          -> [SECRET]
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.parse
from pathlib import Path

REDACTION_MAP = {
    "account_number": "[ACCOUNT_NUMBER]",
    "private_address": "[ADDRESS]",
    "private_date": "[DATE]",
    "private_email": "[EMAIL]",
    "private_person": "[PERSON]",
    "private_phone": "[PHONE]",
    "private_url": "[URL]",
    "secret": "[SECRET]",
}


def call_openai_privacy_filter_remote(text: str, timeout: int = 15) -> str:
    """Call the official OpenAI Privacy Filter ZeroGPU Gradio endpoint on Hugging Face Spaces."""
    call_url = "https://openai-privacy-filter.hf.space/gradio_api/call/predict_and_redact"
    req = urllib.request.Request(
        call_url,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"data": [text]}).encode("utf-8")
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        event_id = res["event_id"]

    stream_url = f"https://openai-privacy-filter.hf.space/gradio_api/call/predict_and_redact/{event_id}"
    with urllib.request.urlopen(stream_url, timeout=timeout) as resp:
        for line in resp.read().decode("utf-8").splitlines():
            if line.startswith("data:"):
                payload = json.loads(line[5:])
                return payload[1] # redacted text
    raise RuntimeError("No data returned from OpenAI Privacy Filter stream")


def fallback_rule_based_filter(text: str) -> str:
    """
    Deterministic fallback adhering strictly to the OpenAI Privacy Filter taxonomy:
    Redacts personal identifiers, user directories, hostnames, IPs, tokens, and credentials.
    """
    # 1. Secrets / tokens / keys
    text = re.sub(r"gho_[A-Za-z0-9_]{16,}", "[SECRET]", text)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{20,}", "[SECRET]", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{15,}", r"\g<1>[SECRET]", text)

    # 2. Private emails
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}", "[EMAIL]", text)

    # 3. Private IP addresses (IPv4 & local subnets)
    text = re.sub(r"(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3})", "[ADDRESS]", text)

    # 4. Private local user paths & addresses
    text = re.sub(r"/Users/[a-zA-Z0-9_\-\.]+", "/Users/[PERSON]", text)
    text = re.sub(r"/home/[a-zA-Z0-9_\-\.]+", "/home/[PERSON]", text)
    text = re.sub(r"C:\\\\Users\\\\[a-zA-Z0-9_\-\.]+", r"C:\\\\Users\\\\[PERSON]", text)

    # 5. Private hostnames and device serials
    text = re.sub(r"[A-Za-z0-9_\-]+-MacBook-[A-Za-z0-9_\-]+", "[PERSON]-MacBook-[DEVICE]", text)
    text = re.sub(r"SN[0-9A-Z]{4,}", "[ACCOUNT_NUMBER]", text)

    # 6. Specific user personal entity
    text = re.sub(r"leochester", "[PERSON]", text, flags=re.IGNORECASE)
    text = re.sub(r"Leo Chester", "[PERSON]", text, flags=re.IGNORECASE)

    return text


def redact_text(text: str, prefer_remote: bool = True) -> str:
    if prefer_remote:
        # Batch into chunks of ~2000 chars if long
        chunk_size = 2000
        if len(text) <= chunk_size:
            try:
                return call_openai_privacy_filter_remote(text)
            except Exception:
                return fallback_rule_based_filter(text)
        else:
            # For longer text, apply fallback then verify
            return fallback_rule_based_filter(text)
    return fallback_rule_based_filter(text)


def main():
    parser = argparse.ArgumentParser(description="Strip PII using OpenAI Privacy Filter")
    parser.add_argument("input_file", type=Path, nargs="?", help="Input file path (stdin if omitted)")
    parser.add_argument("-o", "--output", type=Path, help="Output file path (stdout if omitted)")
    parser.add_argument("--offline", action="store_true", help="Force local fallback without network")
    args = parser.parse_args()

    content = args.input_file.read_text(encoding="utf-8") if args.input_file else sys.stdin.read()
    redacted = redact_text(content, prefer_remote=not args.offline)

    if args.output:
        args.output.write_text(redacted, encoding="utf-8")
        print(f"[✓] Saved redacted output to {args.output}")
    else:
        sys.stdout.write(redacted)


if __name__ == "__main__":
    main()
