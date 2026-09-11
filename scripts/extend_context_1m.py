#!/usr/bin/env python3
"""
Extend Edge0-35B (or Qwen-based hybrid models) context window to 1M tokens (1,048,576)
using YaRN (Yet another RoPE extensioN) positional embedding scaling.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def extend_config_to_1m(config_path: Path, target_context: int = 1048576):
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    # Backup original
    backup_path = config_path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"[+] Created backup at {backup_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle nested text_config (e.g. for MoE / VL architectures)
    target_obj = data.get("text_config", data)
    orig_max = target_obj.get("max_position_embeddings", 262144)
    if orig_max > target_context:
        print(f"[*] max_position_embeddings is already {orig_max} >= {target_context}")
        return

    scale_factor = float(target_context) / float(orig_max)
    print(f"[*] Scaling context from {orig_max:,} -> {target_context:,} (factor: {scale_factor:.2f})")

    target_obj["max_position_embeddings"] = target_context

    yarn_parameters = {
        "type": "yarn",
        "factor": scale_factor,
        "original_max_position_embeddings": orig_max,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "partial_rotary_factor": target_obj.get("partial_rotary_factor", 0.25),
        "rope_theta": target_obj.get("rope_theta", 10000000),
    }

    target_obj["rope_parameters"] = yarn_parameters
    target_obj["rope_scaling"] = yarn_parameters

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[✓] Successfully updated {config_path} to {target_context:,} tokens with YaRN RoPE.")


def main():
    parser = argparse.ArgumentParser(description="Extend model context window to 1M tokens with YaRN")
    parser.add_argument("model_path", type=Path, help="Path to model directory or config.json")
    parser.add_argument("--tokens", type=int, default=1048576, help="Target context window tokens (default: 1,048,576)")
    args = parser.parse_args()

    cfg_path = args.model_path if args.model_path.is_file() else args.model_path / "config.json"
    extend_config_to_1m(cfg_path, target_context=args.tokens)


if __name__ == "__main__":
    main()
