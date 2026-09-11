#!/usr/bin/env bash
# ==============================================================================
# setup_claude_code.sh
# Configure environment variables and shell aliases for Claude Code integration
# with Edge0-35B (1M Context) running locally on port 8000.
# ==============================================================================

set -euo pipefail

ENV_BLOCK='
# --- Edge0 Local LLM & Claude Code (1M Context) ---
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="local"
export ANTHROPIC_API_KEY=""
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="1048576"
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1
alias claude="command claude --model edge0-35b"
alias opencode="OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=dummy opencode --model edge0-35b"
# --------------------------------------------------
'

append_if_missing() {
    local target="$1"
    if [ -f "$target" ]; then
        if ! grep -q "CLAUDE_CODE_MAX_CONTEXT_TOKENS" "$target"; then
            echo "$ENV_BLOCK" >> "$target"
            echo "[+] Added Edge0 & Claude Code environment to $target"
        else
            echo "[*] $target already configured"
        fi
    fi
}

append_if_missing "$HOME/.bashrc"
append_if_missing "$HOME/.zshrc"

echo "[✓] Setup complete! Reload your shell with:"
echo "    source ~/.bashrc   # or source ~/.zshrc"
echo ""
echo "Then launch Claude Code directly:"
echo "    claude"
