#!/usr/bin/env python3
"""Parse backend_parity_microbench CSV and compute SLO ratios."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    text = args.csv_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line or line.startswith("backend") or "Compiling" in line:
            continue
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            backend, op, mode, med = parts[0], parts[1], parts[2], float(parts[3])
        except ValueError:
            continue
        rows.append({"backend": backend, "op": op, "mode": mode, "median_ms": med})

    by = defaultdict(dict)
    for r in rows:
        if r["mode"].startswith("batch") or r["mode"] == "sync":
            by[(r["op"], r["mode"])][r["backend"]] = r["median_ms"]

    violations = []
    table = []
    hot = []
    for (op, mode), m in sorted(by.items()):
        if "cuda" not in m:
            continue
        cuda = m["cuda"]
        rec = {"op": op, "mode": mode, "cuda_ms": cuda}
        for b in ("vulkan", "wgpu"):
            if b in m and cuda > 0:
                ratio = m[b] / cuda
                rec[f"{b}_ms"] = m[b]
                rec[f"{b}_vs_cuda"] = ratio
                # SLO rules
                if b == "vulkan" and ratio > 1.5:
                    violations.append(
                        {"backend": b, "op": op, "mode": mode, "ratio": ratio, "limit": 1.5}
                    )
                if b == "wgpu" and ratio > 2.0:
                    violations.append(
                        {"backend": b, "op": op, "mode": mode, "ratio": ratio, "limit": 2.0}
                    )
                if mode.startswith("batch") and "matmul" in op:
                    hot.append((b, ratio))
        table.append(rec)

    # geomean of hot matmul batch ratios
    def geomean(pairs, backend):
        vals = [r for b, r in pairs if b == backend]
        if not vals:
            return None
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    report = {
        "table": table,
        "violations": violations,
        "vulkan_matmul_batch_geomean_vs_cuda": geomean(hot, "vulkan"),
        "wgpu_matmul_batch_geomean_vs_cuda": geomean(hot, "wgpu"),
        "vulkan_slo_critical_kernel": 1.5,
        "wgpu_slo_critical_kernel": 2.0,
        "e2e_note": "Transformer/LLM e2e not executed in this pack — see STATUS.md",
    }
    out = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(out + "\n", encoding="utf-8")
    print(out)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
