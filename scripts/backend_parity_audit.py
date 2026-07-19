#!/usr/bin/env python3
"""
Static audit of CUDA vs Vulkan/WebGPU BackendStorage method surface.

Scans:
  - candle-core/src/backend.rs          (trait method names)
  - candle-core/src/cuda_backend/mod.rs
  - candle-core/src/vulkan_backend.rs
  - candle-core/src/wgpu_backend.rs

Exits non-zero if CUDA implements a BackendStorage compute/storage method
and BOTH Vulkan and WebGPU either lack an impl or only immediately return
unsupported/not-implemented for that method.

Usage:
  python scripts/backend_parity_audit.py
  python scripts/backend_parity_audit.py --repo /path/to/candle
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# Methods that are accessors / host bridges, not GPU compute parity targets.
ACCESSOR_OPS = frozenset({"dtype", "device"})

# CUDA intentionally does not implement these (trait default or explicit bail).
CUDA_UNSUPPORTED_OPS = frozenset(
    {
        "cumsum_last_dim",  # BackendStorage default: not implemented
        "clamp",  # BackendStorage default: not implemented
        "upsample_nearest1d",  # cuda_backend bail
    }
)

# Regex: fn name at start of a method signature inside impl blocks.
FN_SIG_RE = re.compile(
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?fn\s+([A-Za-z0-9_]+)\s*(?:<[^>]*>)?\s*\("
)

# Immediate unsupported / not-implemented patterns in the first ~40 non-empty lines of a body.
IMMEDIATE_FAIL_RES = [
    re.compile(r"return\s+Err\(\s*unsupported\s*\("),
    re.compile(r"return\s+Err\(\s*Error::UnsupportedDTypeForOp"),
    re.compile(r'bail!\(\s*"upsample-nearest1d is not supported'),
    re.compile(r'Error::Msg\(\s*format!\(\s*"\w+ backend op \{op\} not implemented"'),
    re.compile(r'Err\(crate::Error::Msg\(\s*"backend op .+ not implemented"'),
    re.compile(r'crate::bail!\(\s*"upsample-nearest1d is not supported'),
]


@dataclass
class MethodInfo:
    name: str
    start_line: int  # 1-based
    body: str
    immediately_unsupported: bool = False


@dataclass
class BackendScan:
    path: Path
    methods: Dict[str, MethodInfo] = field(default_factory=dict)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_trait_methods(backend_rs: str, trait_name: str) -> List[str]:
    """Extract method names declared on a trait (including default bodies)."""
    # Find trait block roughly.
    m = re.search(rf"pub\s+trait\s+{trait_name}\b[^{{]*\{{", backend_rs)
    if not m:
        return []
    start = m.end()
    # Brace match
    depth = 1
    i = start
    while i < len(backend_rs) and depth:
        c = backend_rs[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    body = backend_rs[start : i - 1]
    names = []
    for fm in FN_SIG_RE.finditer(body):
        names.append(fm.group(1))
    return names


def find_impl_blocks(src: str, type_name: str) -> List[Tuple[int, str]]:
    """
    Return list of (start_line, block_source) for
      impl ... for TypeName { ... }
    """
    blocks: List[Tuple[int, str]] = []
    # Match impl ... for CudaStorage / VulkanStorage / WgpuStorage
    pattern = re.compile(
        rf"(?m)^impl\b[^{{]*\bfor\s+{type_name}\s*(?:where[^{{]*)?\{{"
    )
    for m in pattern.finditer(src):
        start_idx = m.end()
        start_line = src.count("\n", 0, m.start()) + 1
        depth = 1
        i = start_idx
        while i < len(src) and depth:
            c = src[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        block = src[start_idx : i - 1]
        blocks.append((start_line, block))
    return blocks


def split_methods(block: str, block_start_line: int) -> Dict[str, MethodInfo]:
    """Parse fn methods from an impl block body."""
    methods: Dict[str, MethodInfo] = {}
    # Find each fn and take until next fn at same indent or end.
    matches = list(FN_SIG_RE.finditer(block))
    for idx, m in enumerate(matches):
        name = m.group(1)
        # Skip nested helpers that are clearly not trait methods? Keep all; filter later.
        body_start = m.start()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
        body = block[body_start:body_end]
        line = block_start_line + block[: m.start()].count("\n")
        methods[name] = MethodInfo(
            name=name,
            start_line=line,
            body=body,
            immediately_unsupported=is_immediately_unsupported(body),
        )
    return methods


def is_immediately_unsupported(method_src: str) -> bool:
    """
    True if the method body has no real GPU work and fails immediately.

    Heuristic: look at the body after the opening brace of the fn; if within
    the first meaningful statements we only see return Err(unsupported...) /
    not-implemented, without intermediate GPU helpers.
    """
    # Extract first brace body
    brace = method_src.find("{")
    if brace < 0:
        return False
    body = method_src[brace + 1 :]
    # Collapse to first 50 non-empty, non-comment lines
    lines: List[str] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        # skip attribute-like
        if s.startswith("#["):
            continue
        lines.append(s)
        if len(lines) >= 50:
            break
    snippet = "\n".join(lines[:25])

    # CUDA upsample_nearest1d style
    if "upsample-nearest1d is not supported" in snippet:
        return True

    # Pure unsupported return as first real statement(s)
    # Allow let bindings that only bind self fields before immediate Err.
    first_exec = None
    for s in lines[:15]:
        if s.startswith("let ") or s.startswith("use ") or s.startswith("const "):
            continue
        if s.startswith("if ") or s.startswith("match ") or s.startswith("return ") or s.startswith("Err("):
            first_exec = s
            break
        # assignment / call — real work
        first_exec = s
        break

    if first_exec is None:
        return False

    joined = "\n".join(lines[:15])
    full_joined = "\n".join(lines[:50])
    # Real GPU work signals — edge-case UnsupportedDType early-returns must not
    # mark a fully implemented method as missing.
    gpuish = re.search(
        r"\b("
        r"run_|self\.(run_|materialize|bf16_|cuda_parity)|get_or_load_func|"
        r"LaunchConfig|gemm_|copy_buffer|BufferCopy|submit_copy|create_command|"
        r"dispatch|encoder\.|queue\.|vk::|ash::|wgpu::"
        r")",
        full_joined,
    )
    for rx in IMMEDIATE_FAIL_RES:
        if rx.search(joined) and not gpuish:
            return True

    # Trait default style single-line not implemented (whole body is just Err)
    if re.search(r'backend op .+ not implemented', joined) and not gpuish:
        return True

    return False


def scan_backend(path: Path, type_name: str) -> BackendScan:
    src = read_text(path)
    scan = BackendScan(path=path)
    for start_line, block in find_impl_blocks(src, type_name):
        for name, info in split_methods(block, start_line).items():
            # Prefer first occurrence (trait impl usually one big block)
            if name not in scan.methods:
                scan.methods[name] = info
    return scan


def classify_status(info: Optional[MethodInfo], cuda_unsupported: bool) -> str:
    if cuda_unsupported:
        return "unsupported_spec"
    if info is None:
        return "missing"
    if info.immediately_unsupported:
        return "missing"
    # Heuristic partial vs native is not reliable from static scan alone;
    # report as "present" for summary; JSON manifest holds curated native/partial.
    return "present"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Candle repo root (default: parent of scripts/)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to docs/backend-parity-manifest.json for status counts",
    )
    args = ap.parse_args()
    root = (args.repo or repo_root_from_script()).resolve()

    backend_rs = root / "candle-core" / "src" / "backend.rs"
    cuda_mod = root / "candle-core" / "src" / "cuda_backend" / "mod.rs"
    vulkan_rs = root / "candle-core" / "src" / "vulkan_backend.rs"
    wgpu_rs = root / "candle-core" / "src" / "wgpu_backend.rs"

    for p in (backend_rs, cuda_mod, vulkan_rs, wgpu_rs):
        if not p.is_file():
            print(f"ERROR: missing required file: {p}", file=sys.stderr)
            return 2

    trait_src = read_text(backend_rs)
    storage_ops = extract_trait_methods(trait_src, "BackendStorage")
    device_ops = extract_trait_methods(trait_src, "BackendDevice")

    if not storage_ops:
        print("ERROR: failed to parse BackendStorage methods", file=sys.stderr)
        return 2

    cuda = scan_backend(cuda_mod, "CudaStorage")
    # Device methods live in device.rs
    cuda_dev_path = root / "candle-core" / "src" / "cuda_backend" / "device.rs"
    cuda_dev = scan_backend(cuda_dev_path, "CudaDevice") if cuda_dev_path.is_file() else BackendScan(cuda_dev_path)

    vulkan = scan_backend(vulkan_rs, "VulkanStorage")
    vulkan_dev = scan_backend(vulkan_rs, "VulkanDevice")
    wgpu = scan_backend(wgpu_rs, "WgpuStorage")
    wgpu_dev = scan_backend(wgpu_rs, "WgpuDevice")

    # CUDA has method if present in scan OR not in CUDA_UNSUPPORTED (defaults count as "no cuda impl")
    def cuda_has_real(op: str) -> bool:
        if op in CUDA_UNSUPPORTED_OPS:
            return False
        info = cuda.methods.get(op)
        if info is None:
            return False
        return not info.immediately_unsupported

    failures: List[str] = []
    rows = []

    for op in storage_ops:
        if op in ACCESSOR_OPS:
            continue
        c_info = cuda.methods.get(op)
        v_info = vulkan.methods.get(op)
        w_info = wgpu.methods.get(op)

        c_real = cuda_has_real(op)
        # If CUDA uses trait default (no override), treat as not required
        if op in CUDA_UNSUPPORTED_OPS or c_info is None:
            # double-check upsample etc. that exist but bail
            if c_info is not None and c_info.immediately_unsupported:
                c_real = False
            elif op in CUDA_UNSUPPORTED_OPS:
                c_real = False
            elif c_info is None and op in ("cumsum_last_dim", "clamp"):
                c_real = False

        v_ok = v_info is not None and not v_info.immediately_unsupported
        w_ok = w_info is not None and not w_info.immediately_unsupported

        status = {
            "op": op,
            "cuda": "yes" if c_real else ("unsupported" if op in CUDA_UNSUPPORTED_OPS or (c_info and c_info.immediately_unsupported) else "missing"),
            "vulkan": "yes" if v_ok else ("missing" if v_info is None else "immediate_unsupported"),
            "wgpu": "yes" if w_ok else ("missing" if w_info is None else "immediate_unsupported"),
            "cuda_line": c_info.start_line if c_info else None,
            "vulkan_line": v_info.start_line if v_info else None,
            "wgpu_line": w_info.start_line if w_info else None,
        }
        rows.append(status)

        if c_real and (not v_ok) and (not w_ok):
            failures.append(
                f"CUDA op '{op}' has no usable Vulkan/WGPU impl "
                f"(vulkan={status['vulkan']}, wgpu={status['wgpu']})"
            )
        elif c_real and not v_ok:
            # Soft warn: still exit non-zero only if BOTH missing (per task).
            # Task: "without a corresponding impl that doesn't immediately return unsupported for both"
            pass
        elif c_real and not w_ok:
            pass

    # Device surface (informational + fail if zeros/alloc/upload missing on both)
    critical_device = {
        "zeros_impl",
        "alloc_uninit",
        "storage_from_cpu_storage",
        "storage_from_cpu_storage_owned",
        "synchronize",
    }
    for op in device_ops:
        c_info = cuda_dev.methods.get(op)
        v_info = vulkan_dev.methods.get(op)
        w_info = wgpu_dev.methods.get(op)
        c_real = c_info is not None and not c_info.immediately_unsupported
        v_ok = v_info is not None and not v_info.immediately_unsupported
        w_ok = w_info is not None and not w_info.immediately_unsupported
        if op in critical_device and c_real and (not v_ok) and (not w_ok):
            failures.append(
                f"CUDA device op '{op}' missing usable Vulkan/WGPU impl"
            )

    # Summary counts from static presence
    present_v = sum(1 for r in rows if r["vulkan"] == "yes")
    present_w = sum(1 for r in rows if r["wgpu"] == "yes")
    miss_v = sum(1 for r in rows if r["vulkan"] != "yes")
    miss_w = sum(1 for r in rows if r["wgpu"] != "yes")
    cuda_req = sum(1 for r in rows if r["cuda"] == "yes")

    # Optional curated manifest counts + schema v2 validation
    ALLOWED_STATUSES = {
        "Native",
        "Optimized",
        "GPUEmulated",
        "UnsupportedBySpecification",
        "UnsupportedByHardware",
        "CudaSpecific",
        "Missing",
        "Verified",
    }
    PROFILE_FIELDS = (
        "vulkan_status",
        "native_webgpu_status",
        "portable_webgpu_status",
    )
    manifest_counts = None
    manifest_path = args.manifest or (root / "docs" / "backend-parity-manifest.json")
    # Release gate: curated three-profile manifest is required (not optional soft-pass).
    if not manifest_path.is_file():
        failures.append(
            f"required parity manifest missing: {manifest_path} "
            "(create docs/backend-parity-manifest.json schema_version>=2)"
        )
    else:
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "ops" in raw:
                data = raw["ops"]
                schema_version = raw.get("schema_version", 1)
            else:
                data = raw
                schema_version = 1

            if schema_version < 2:
                failures.append(
                    f"parity manifest schema_version={schema_version} < 2 "
                    "(require three profile fields per op)"
                )

            def count_status(key: str) -> Dict[str, int]:
                out: Dict[str, int] = {}
                for item in data:
                    st = item.get(key, "unknown")
                    out[st] = out.get(st, 0) + 1
                return out

            # Enforce vocabulary + three profiles on schema v2 (and soft on v1 if fields present)
            for item in data:
                op_name = item.get("op", "<unknown>")
                for field in PROFILE_FIELDS:
                    if field not in item:
                        if schema_version >= 2:
                            failures.append(
                                f"manifest op '{op_name}' missing required profile field '{field}'"
                            )
                        continue
                    st = item[field]
                    if st not in ALLOWED_STATUSES:
                        failures.append(
                            f"manifest op '{op_name}' field '{field}' has unknown status {st!r}"
                        )
                    # Unexplained Missing is incomplete work for release-required CUDA surface.
                    if st == "Missing":
                        # Meta/edge rows may still use Missing only if explicitly documented;
                        # CUDA-required BackendStorage ops must never remain Missing.
                        if op_name in {r["op"] for r in rows if r["cuda"] == "yes"}:
                            failures.append(
                                f"manifest op '{op_name}' has unexplained Missing on {field} "
                                "(required CUDA surface must be implemented or classified)"
                            )
                    if st == "Verified" and not item.get("tests"):
                        # policy/meta rows may use Verified with structural evidence
                        if not str(op_name).startswith("edge::") and op_name not in (
                            "cpu_fallback_policy",
                        ):
                            failures.append(
                                f"manifest op '{op_name}' is Verified on {field} but has no tests[]"
                            )
                    # Portable Verified must not be claimed from native-only smoke evidence.
                    if field == "portable_webgpu_status" and st == "Verified":
                        portable_tests = item.get("portable_tests") or []
                        tests = item.get("tests") or []
                        banned_native_only = (
                            "backend_smoke_tests",
                            "gpu_parity_matrix",
                            "gpu_property_tests",
                            "gpu_metamorphic_tests",
                        )

                        def is_portable_evidence(path: object) -> bool:
                            s = str(path).lower()
                            if any(b in s for b in banned_native_only):
                                return False
                            return any(
                                tag in s
                                for tag in (
                                    "wasm",
                                    "portable",
                                    "browser",
                                    "candle-wasm",
                                )
                            )

                        # Policy rows may use structural evidence without a harness.
                        if op_name in ("cpu_fallback_policy",):
                            has_portable_evidence = bool(portable_tests or tests)
                        else:
                            has_portable_evidence = any(
                                is_portable_evidence(t) for t in portable_tests
                            ) or any(is_portable_evidence(t) for t in tests)
                        if not has_portable_evidence:
                            failures.append(
                                f"manifest op '{op_name}' portable_webgpu_status is Verified "
                                "but lacks portable/WASM/browser test evidence "
                                "(native wgpu smokes are not portable Verified)"
                            )
                    if st == "Optimized" and not item.get("bench"):
                        failures.append(
                            f"manifest op '{op_name}' is Optimized on {field} but bench is false"
                        )

            # Every CUDA-required storage op must appear in curated manifest
            manifest_ops = {item.get("op") for item in data}
            for r in rows:
                if r["cuda"] == "yes" and r["op"] not in manifest_ops:
                    failures.append(
                        f"CUDA-required op '{r['op']}' missing from parity manifest"
                    )

            # Critical device ops must appear too
            for dop in critical_device:
                if dop not in manifest_ops:
                    failures.append(
                        f"critical device op '{dop}' missing from parity manifest"
                    )

            manifest_counts = {
                "schema_version": schema_version,
                "vulkan": count_status("vulkan_status"),
                "native_webgpu": count_status("native_webgpu_status")
                if data and "native_webgpu_status" in data[0]
                else count_status("wgpu_status"),
                "portable_webgpu": count_status("portable_webgpu_status")
                if data and "portable_webgpu_status" in data[0]
                else {},
                "wgpu": count_status("wgpu_status"),
                "rows": len(data),
            }
        except Exception as e:  # noqa: BLE001
            print(f"WARN: could not load manifest: {e}", file=sys.stderr)
            failures.append(f"manifest load error: {e}")

    if args.json:
        payload = {
            "cuda_required_ops": cuda_req,
            "rows": rows,
            "failures": failures,
            "static_presence": {
                "vulkan_present": present_v,
                "wgpu_present": present_w,
                "vulkan_not_present": miss_v,
                "wgpu_not_present": miss_w,
            },
            "manifest_counts": manifest_counts,
        }
        print(json.dumps(payload, indent=2))
    else:
        print("=== Candle backend parity audit (static) ===")
        print(f"repo: {root}")
        print(f"BackendStorage ops in trait: {len(storage_ops)}")
        print(f"CUDA-required (real impl):   {cuda_req}")
        print()
        print(f"{'op':28} {'cuda':12} {'vulkan':22} {'wgpu':22}")
        print("-" * 90)
        for r in rows:
            print(
                f"{r['op']:28} {r['cuda']:12} {r['vulkan']:22} {r['wgpu']:22}"
            )
        print()
        print("Static presence (not curated native/partial):")
        print(f"  Vulkan present: {present_v}  not-present/unsupported: {miss_v}")
        print(f"  WGPU   present: {present_w}  not-present/unsupported: {miss_w}")
        if manifest_counts:
            print()
            print(f"Curated manifest ({manifest_path}): {manifest_counts['rows']} rows")
            print(f"  schema: {manifest_counts.get('schema_version')}")
            print(f"  Vulkan:          {manifest_counts['vulkan']}")
            print(f"  Native WebGPU:   {manifest_counts.get('native_webgpu')}")
            print(f"  Portable WebGPU: {manifest_counts.get('portable_webgpu')}")
        print()
        if failures:
            print(f"FAILURES ({len(failures)}):")
            for f in failures:
                print(f"  - {f}")
        else:
            print(
                "OK: every CUDA-required BackendStorage op has a non-immediate "
                "impl on Vulkan and/or WGPU (both when required)."
            )
            print(
                "OK: parity manifest uses allowed three-profile statuses "
                "(Native/Optimized/GPUEmulated/UnsupportedBySpecification/"
                "UnsupportedByHardware/CudaSpecific/Missing/Verified)."
            )
            print(
                "Note: depth gaps (dtype/layout/perf) are tracked in "
                "docs/backend-parity-manifest.json / docs/backend-parity.md"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
