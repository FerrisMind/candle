#!/usr/bin/env python3
"""Static audit for forbidden CPU compute / silent cast patterns in GPU backends."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Patterns that suggest compute-oriented host fallback (not intentional D2H).
SUSPECT = [
    (r"to_cpu_storage\s*\(", "to_cpu_storage call"),
    (r"\.to_vec1\s*\(|\.to_vec2\s*\(|\.to_vec3\s*\(", "to_vec host materialize"),
    (r"map_async\s*\(", "map_async"),
    (r"map_buffer\s*\(", "map_buffer"),
    (r"\.read_buffer\s*\(", "read_buffer call"),
    (r"record_.*cpu_fallback\s*\(", "cpu_fallback recorder call"),
    (r"record_.*host_compute\s*\(", "host_compute recorder call"),
]

# Allowlist: intentional host bridges (line content substrings / path rules).
ALLOW_PATH_SNIPPETS = [
    "fn to_cpu_storage",
    "fn transfer_to_device",
    "fn storage_from_cpu",
    "bytes_to_cpu_storage",
    "cpu_storage_to_bytes",
    "fn data(",  # quantized data download
    "fn read_buffer(",  # definition
    "fn record_",  # definition of recorder
    "#[test]",
    "mod tests",
    "// Intentional",
]


def classify_line(path: Path, line_no: int, line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("//"):
        return None
    for allow in ALLOW_PATH_SNIPPETS:
        if allow in line:
            return None
    for rx, name in SUSPECT:
        if re.search(rx, line):
            return name
    return None


def scan(path: Path) -> list[dict]:
    hits = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        kind = classify_line(path, i, line)
        if kind:
            hits.append(
                {
                    "file": str(path).replace("\\", "/"),
                    "line": i,
                    "kind": kind,
                    "text": line.strip()[:200],
                }
            )
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    root = (args.repo or Path(__file__).resolve().parents[1]).resolve()
    files = [
        root / "candle-core/src/vulkan_backend.rs",
        root / "candle-core/src/wgpu_backend.rs",
        root / "candle-core/src/storage.rs",
    ]
    all_hits = []
    for f in files:
        if f.is_file():
            all_hits.extend(scan(f))

    # Separate intentional categories
    intentional = []
    compute_suspects = []
    for h in all_hits:
        if h["kind"] in ("to_cpu_storage call", "CpuStorage construct") and (
            "fn to_cpu" in h["text"]
            or "bytes_to_cpu" in h["text"]
            or "storage_from" in h["text"]
            or "transfer_to" in h["text"]
        ):
            intentional.append(h)
        elif "mod tests" in h.get("text", "") or "/tests/" in h["file"]:
            intentional.append(h)
        else:
            compute_suspects.append(h)

    report = {
        "suspect_count": len(compute_suspects),
        "intentional_or_bridge_count": len(intentional),
        "suspects": compute_suspects[:200],
        "note": (
            "Suspects require manual classification. Runtime STRICT mode uses "
            "CANDLE_STRICT_NO_CPU_FALLBACK=1 with record_* hooks."
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== fallback static audit ===")
        print(f"suspects: {report['suspect_count']}")
        print(f"bridges/intentional-ish: {report['intentional_or_bridge_count']}")
        for h in compute_suspects[:40]:
            print(f"  {h['file']}:{h['line']} [{h['kind']}] {h['text']}")
    # Non-zero only if clearly bad patterns: host recompute after error, etc.
    hard = [
        h
        for h in compute_suspects
        if "record_" in h["kind"] or "to_vec" in h["kind"] and "test" not in h["file"]
    ]
    # to_vec in production backends is bad if not in tests section
    return 0 if len(hard) < 5 else 1


if __name__ == "__main__":
    sys.exit(main())
