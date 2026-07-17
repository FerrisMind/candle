#!/usr/bin/env python3
"""
Lightweight host-side orchestrator for Candle backend microbenches.

Runs the in-repo Rust harness (when present) or records environment metadata
for manual comparison. Prefer the Rust binary for actual GPU timings.

Usage:
  python scripts/backend_microbench.py --env-only
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def collect_env() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "rustc": run(["rustc", "--version"]),
        "cargo": run(["cargo", "--version"]),
        "nvidia_smi": run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": run(["nvcc", "--version"]),
        "glslc": run(["glslc", "--version"]),
        "note": (
            "Run release microbenches via candle-core benches and "
            "candle-nn sdpa_bench with features cuda/vulkan/wgpu."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env-only", action="store_true")
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args()
    env = collect_env()
    text = json.dumps(env, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
