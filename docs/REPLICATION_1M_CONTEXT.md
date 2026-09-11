# Replicating 1M Context & Local Claude Code Integration on Any Device

This guide provides everything required to replicate the **1 Million Token Context Window (1,048,576 tokens)** on Edge0-35B and wire it directly into **Claude Code CLI** (or OpenCode / Cursor / any Anthropic/OpenAI compatible client) on any device.

---

## 1. Why 1M Context Fits on an 8 GB RAM Device

Most 30B+ models cannot exceed 32k–64k context on consumer hardware because standard multi-head self-attention KV cache scales linearly with sequence length ($O(N)$) across all layers. At 1M tokens in fp16, a standard 70B model requires $pprox 64 	ext{ GB}$ of RAM just for KV cache.

**Edge0-35B overcomes this through an interleaved hybrid architecture:**
1. **40 Total Layers**:
   - **30 Recurrent Linear Attention Layers** (Gated DeltaNet):
     - Maintained by constant $O(1)$ state (`ArraysCache(size=2)`).
     - **0 MB memory growth** regardless of sequence length.
   - **10 Full Attention Layers** (`full_attention_interval: 4`):
     - Full attention is evaluated only once every 4 layers.
     - 2 KV heads (Grouped Query Attention) with head dimension 128.
2. **4-Bit Quantized KV Cache (`QuantizedKVCache`)**:
   - Compresses the 10 full-attention layers from 16-bit to 4-bit (`bits=4, group_size=64`).
   - Memory cost: $10 	ext{ layers} 	imes 2 	ext{ KV heads} 	imes 128 	ext{ dim} 	imes 2 	ext{ (K, V)} 	imes 0.5 	ext{ bytes} pprox 2.5 	ext{ KB / token}$.
   - **1,048,576 tokens $	imes 2.5 	ext{ KB} = 1.25 	ext{ GB}$ total KV cache**!
3. **Total Footprint**:
   $$	ext{Active Quantized Weights } (\sim 2.9	ext{ GB}) + 	ext{1M KV Cache } (\sim 1.25	ext{ GB}) pprox \mathbf{4.15	ext{ GB RAM}}$$
   This easily fits inside an 8 GB unified memory budget (e.g. MacBook Neo / M-series Mac).

---

## 2. YaRN RoPE Scaling Parameters

To extend the positional embeddings from 262,144 (256k) to 1,048,576 (1M) without degrading local perplexity, we apply **YaRN (Yet another RoPE extensioN)**:

```json
"rope_parameters": {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000
}
```

- **Scale factor ($s = 4.0$)**: $1{,}048{,}576 / 262{,}144 = 4.0$.
- **$eta_{fast} = 32, eta_{slow} = 1$**: Preserves full high-frequency representation for local context while smoothly interpolating distant tokens.
- **$	ext{rope\_theta} = 10{,}000{,}000$**: Extends base wavelength to prevent rotational wrapping over 1M steps.

---

## 3. Quick Replication (Automated)

### Step 1: Clone the Repository
```bash
git clone https://github.com/TheGitCommitMan/Edge0.git
cd Edge0
```

### Step 2: Set Up Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Step 3: Patch Your Model Config to 1M Context
Run the included replication script pointing to your model checkpoint:
```bash
python3 scripts/extend_context_1m.py models/edge0-35b
```

### Step 4: Launch the Edge0 Daemon
```bash
edge0 serve models/edge0-35b --port 8000
```

### Step 5: Configure Shell for Claude Code
Run the setup script or add to your `~/.bashrc` / `~/.zshrc`:
```bash
bash scripts/setup_claude_code.sh
source ~/.bashrc
```

Now start Claude Code:
```bash
claude
```
Claude Code connects directly to your local Edge0 instance with 1M context unlocked!

---

## 4. Replicating on Other Devices & Accelerators

- **Apple Silicon (M1/M2/M3/M4/A18 Pro)**:
  - Uses the MLX backend natively. Metal kernels automatically execute 4-bit quantized attention.
- **Linux / CUDA (NVIDIA)**:
  - For vLLM / SGLang / HuggingFace backends: pass `rope_scaling={"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}` in `config.json`, and enable `--kv-cache-dtype fp8` or 4-bit KV cache quantization.

---

## 5. Sanitized Agent Trajectory

To understand every decision, debugging phase, and technical derivation, read:
- [`docs/AGENT_CONVERSATION_TRANSCRIPT_PII_STRIPPED.md`](AGENT_CONVERSATION_TRANSCRIPT_PII_STRIPPED.md)
