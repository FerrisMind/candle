#!/usr/bin/env python3
"""
Strict static audit of CUDA vs Vulkan / Native WebGPU BackendStorage surface
and curated three-profile parity manifest.

Hard rules (verification stage):
  * CUDA-required op missing on Vulkan alone OR WGPU alone → FAIL
  * Method presence alone is not enough if body immediately unsupported
  * Detect unconditional and common dtype/layout-dependent unsupported branches
  * Verified requires test *function* references that exist and are not
    over-shared generic files claiming unrelated ops
  * Manifest schema v2 with three profiles is mandatory

Usage:
  python scripts/backend_parity_audit.py
  python scripts/backend_parity_audit.py --repo /path/to/candle --json
  python scripts/backend_parity_audit.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ACCESSOR_OPS = frozenset({"dtype", "device"})

CUDA_UNSUPPORTED_OPS = frozenset(
    {
        "cumsum_last_dim",
        "clamp",
        "upsample_nearest1d",
    }
)

# Generic test files that may not alone verify arbitrary ops.
GENERIC_TEST_FILES = frozenset(
    {
        "backend_smoke_tests.rs",
        "gpu_parity_matrix_tests.rs",
        "gpu_property_tests.rs",
        "gpu_metamorphic_tests.rs",
        "gpu_shader_validation_tests.rs",
    }
)

# Max number of Verified ops that may cite the same bare file without ::function
MAX_GENERIC_FILE_SHARE = 3

FN_SIG_RE = re.compile(
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?fn\s+([A-Za-z0-9_]+)\s*(?:<[^>]*>)?\s*\("
)

IMMEDIATE_FAIL_RES = [
    re.compile(r"return\s+Err\(\s*unsupported\s*\("),
    re.compile(r"return\s+Err\(\s*Error::UnsupportedDTypeForOp"),
    re.compile(r'bail!\(\s*"upsample-nearest1d is not supported'),
    re.compile(r'Error::Msg\(\s*format!\(\s*"\w+ backend op \{op\} not implemented"'),
    re.compile(r'Err\(crate::Error::Msg\(\s*"backend op .+ not implemented"'),
    re.compile(r'crate::bail!\(\s*"upsample-nearest1d is not supported'),
    re.compile(r'return\s+Err\(\s*Error::Msg\(\s*format!\(\s*"wgpu backend op'),
    re.compile(r'return\s+Err\(\s*Error::Msg\(\s*format!\(\s*"vulkan backend op'),
]

# Signals of real GPU work early in method body.
GPUISH_RE = re.compile(
    r"\b("
    r"run_|self\.(run_|materialize|bf16_|cuda_parity)|get_or_load_func|"
    r"LaunchConfig|gemm_|copy_buffer|BufferCopy|submit_copy|create_command|"
    r"dispatch|encoder\.|queue\.|vk::|ash::|wgpu::|"
    r"run_compute|run_unary|run_binary|run_matmul|run_im2col|"
    r"copy_strided|const_set|to_dtype"
    r")"
)

# Dtype/layout-dependent unsupported branches (for reporting / soft flags).
DTYPE_FAIL_RE = re.compile(
    r"UnsupportedDTypeForOp|unsupported\(|backend op .+ not implemented|"
    r"not implemented for|requires shader-f16|requires SHADER_F16|"
    r"rank > 4|supports up to rank"
)

ALLOWED_STATUSES = {
    "Native",
    "Optimized",
    "GPUEmulated",
    "UnsupportedBySpecification",
    "UnsupportedByHardware",
    "CudaSpecific",
    "Missing",
    "Verified",
    "Unverified",  # explicit verification-stage status for unproven claims
}

PROFILE_FIELDS = (
    "vulkan_status",
    "native_webgpu_status",
    "portable_webgpu_status",
)


@dataclass
class MethodInfo:
    name: str
    start_line: int
    body: str
    immediately_unsupported: bool = False
    dtype_dependent_unsupported: bool = False
    has_gpu_work: bool = False


@dataclass
class BackendScan:
    path: Path
    methods: Dict[str, MethodInfo] = field(default_factory=dict)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_trait_methods(backend_rs: str, trait_name: str) -> List[str]:
    m = re.search(rf"pub\s+trait\s+{trait_name}\b[^{{]*\{{", backend_rs)
    if not m:
        return []
    start = m.end()
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
    return [fm.group(1) for fm in FN_SIG_RE.finditer(body)]


def find_impl_blocks(src: str, type_name: str) -> List[Tuple[int, str]]:
    blocks: List[Tuple[int, str]] = []
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
        blocks.append((start_line, src[start_idx : i - 1]))
    return blocks


def is_immediately_unsupported(method_src: str) -> bool:
    brace = method_src.find("{")
    if brace < 0:
        return False
    body = method_src[brace + 1 :]
    lines: List[str] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        if s.startswith("#["):
            continue
        lines.append(s)
        if len(lines) >= 50:
            break
    if not lines:
        return False
    full_joined = "\n".join(lines[:50])
    joined = "\n".join(lines[:15])
    gpuish = GPUISH_RE.search(full_joined)
    for rx in IMMEDIATE_FAIL_RES:
        if rx.search(joined) and not gpuish:
            return True
    if re.search(r"backend op .+ not implemented", joined) and not gpuish:
        return True
    return False


def analyze_method(name: str, start_line: int, body: str) -> MethodInfo:
    imm = is_immediately_unsupported(body)
    brace = body.find("{")
    blob = body[brace + 1 :] if brace >= 0 else body
    has_gpu = bool(GPUISH_RE.search(blob))
    dtype_dep = bool(DTYPE_FAIL_RE.search(blob)) and has_gpu
    # Unconditional: whole body is only unsupported returns / match arms that all fail
    if not imm and not has_gpu and "unsupported(" in blob and "return Err" in blob:
        imm = True
    return MethodInfo(
        name=name,
        start_line=start_line,
        body=body,
        immediately_unsupported=imm,
        dtype_dependent_unsupported=dtype_dep,
        has_gpu_work=has_gpu and not imm,
    )


def split_methods(block: str, block_start_line: int) -> Dict[str, MethodInfo]:
    methods: Dict[str, MethodInfo] = {}
    matches = list(FN_SIG_RE.finditer(block))
    for idx, m in enumerate(matches):
        name = m.group(1)
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
        body = block[m.start() : body_end]
        line = block_start_line + block[: m.start()].count("\n")
        if name not in methods:
            methods[name] = analyze_method(name, line, body)
    return methods


def scan_backend(path: Path, type_name: str) -> BackendScan:
    src = read_text(path)
    scan = BackendScan(path=path)
    for start_line, block in find_impl_blocks(src, type_name):
        for name, info in split_methods(block, start_line).items():
            if name not in scan.methods:
                scan.methods[name] = info
    return scan


def usable(info: Optional[MethodInfo]) -> bool:
    """Presence alone is insufficient: need non-immediate-fail and GPU work signal."""
    if info is None:
        return False
    if info.immediately_unsupported:
        return False
    # Allow host bridges that are intentional (to_cpu_storage) without GPU dispatch
    if info.name in ("to_cpu_storage", "try_clone", "dtype", "device"):
        return True
    return info.has_gpu_work


def list_rust_test_functions(repo: Path) -> Dict[str, Set[str]]:
    """Map relative path -> set of fn names (test and helper)."""
    out: Dict[str, Set[str]] = {}
    tests_root = repo / "candle-core" / "tests"
    if not tests_root.is_dir():
        return out
    for path in tests_root.rglob("*.rs"):
        rel = str(path.relative_to(repo)).replace("\\", "/")
        text = read_text(path)
        names = set(FN_SIG_RE.findall(text))
        out[rel] = names
        # also bare filename key
        out[path.name] = out.get(path.name, set()) | names
    return out


def parse_test_ref(ref: str) -> Tuple[str, Optional[str]]:
    """
    Accept:
      candle-core/tests/foo.rs
      candle-core/tests/foo.rs::bar
      foo.rs::bar
    """
    ref = ref.strip()
    if "::" in ref:
        file_part, fn = ref.rsplit("::", 1)
        return file_part.replace("\\", "/"), fn
    return ref.replace("\\", "/"), None


def validate_test_refs(
    item: dict,
    test_index: Dict[str, Set[str]],
    repo: Path,
    failures: List[str],
    file_claim_count: Dict[str, int],
) -> None:
    op_name = item.get("op", "<unknown>")
    statuses = [
        item.get("vulkan_status"),
        item.get("native_webgpu_status"),
    ]
    if not any(s == "Verified" for s in statuses):
        return
    if str(op_name).startswith("edge::") or op_name in ("cpu_fallback_policy",):
        return

    tests = item.get("tests") or []
    test_functions = item.get("test_functions") or []
    refs = list(tests) + list(test_functions)
    if not refs:
        failures.append(
            f"manifest op '{op_name}' is Verified but has empty tests[]/test_functions[]"
        )
        return

    # Prefer explicit test_functions; bare file refs are limited.
    has_function_ref = False
    for ref in refs:
        file_part, fn = parse_test_ref(str(ref))
        # Resolve index keys
        candidates = [file_part, Path(file_part).name]
        known = None
        for c in candidates:
            if c in test_index:
                known = c
                break
            # try under candle-core/tests/
            alt = f"candle-core/tests/{Path(file_part).name}"
            if alt in test_index:
                known = alt
                break
        if known is None:
            # file may live under examples
            p = repo / file_part
            if p.is_file():
                names = set(FN_SIG_RE.findall(read_text(p)))
                test_index[file_part] = names
                known = file_part
            else:
                failures.append(
                    f"manifest op '{op_name}' test ref '{ref}' file not found"
                )
                continue
        if fn:
            has_function_ref = True
            if fn not in test_index[known]:
                failures.append(
                    f"manifest op '{op_name}' test function '{fn}' not found in {known}"
                )
        else:
            bare = Path(known).name
            if bare in GENERIC_TEST_FILES:
                file_claim_count[bare] = file_claim_count.get(bare, 0) + 1

    if not has_function_ref:
        # Verified requires at least one function-level reference for non-meta ops
        failures.append(
            f"manifest op '{op_name}' is Verified but lacks test function reference "
            f"(use path::function; bare generic file is insufficient)"
        )


def run_audit(root: Path, manifest_path: Optional[Path], as_json: bool) -> int:
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
    cuda_dev_path = root / "candle-core" / "src" / "cuda_backend" / "device.rs"
    cuda_dev = (
        scan_backend(cuda_dev_path, "CudaDevice")
        if cuda_dev_path.is_file()
        else BackendScan(cuda_dev_path)
    )
    vulkan = scan_backend(vulkan_rs, "VulkanStorage")
    vulkan_dev = scan_backend(vulkan_rs, "VulkanDevice")
    wgpu = scan_backend(wgpu_rs, "WgpuStorage")
    wgpu_dev = scan_backend(wgpu_rs, "WgpuDevice")

    def cuda_has_real(op: str) -> bool:
        if op in CUDA_UNSUPPORTED_OPS:
            return False
        info = cuda.methods.get(op)
        if info is None:
            return False
        return not info.immediately_unsupported

    failures: List[str] = []
    warnings: List[str] = []
    rows = []

    for op in storage_ops:
        if op in ACCESSOR_OPS:
            continue
        c_info = cuda.methods.get(op)
        v_info = vulkan.methods.get(op)
        w_info = wgpu.methods.get(op)

        c_real = cuda_has_real(op)
        if op in CUDA_UNSUPPORTED_OPS or (
            c_info is not None and c_info.immediately_unsupported
        ):
            c_real = False
        elif c_info is None and op in ("cumsum_last_dim", "clamp"):
            c_real = False

        v_ok = usable(v_info)
        w_ok = usable(w_info)

        status = {
            "op": op,
            "cuda": "yes"
            if c_real
            else (
                "unsupported"
                if op in CUDA_UNSUPPORTED_OPS
                or (c_info and c_info.immediately_unsupported)
                else "missing"
            ),
            "vulkan": "yes"
            if v_ok
            else ("missing" if v_info is None else "immediate_unsupported"),
            "wgpu": "yes"
            if w_ok
            else ("missing" if w_info is None else "immediate_unsupported"),
            "vulkan_dtype_dependent_unsupported": bool(
                v_info and v_info.dtype_dependent_unsupported
            ),
            "wgpu_dtype_dependent_unsupported": bool(
                w_info and w_info.dtype_dependent_unsupported
            ),
            "cuda_line": c_info.start_line if c_info else None,
            "vulkan_line": v_info.start_line if v_info else None,
            "wgpu_line": w_info.start_line if w_info else None,
        }
        rows.append(status)

        # HARD: fail per-backend, not only when both missing
        if c_real and not v_ok:
            failures.append(
                f"CUDA op '{op}' has no usable Native Vulkan impl "
                f"(status={status['vulkan']}, line={status['vulkan_line']})"
            )
        if c_real and not w_ok:
            failures.append(
                f"CUDA op '{op}' has no usable Native WebGPU impl "
                f"(status={status['wgpu']}, line={status['wgpu_line']})"
            )
        if c_real and v_info and v_info.dtype_dependent_unsupported:
            warnings.append(
                f"Vulkan '{op}' has dtype/layout-dependent unsupported branches "
                f"(line {v_info.start_line}) — classify in manifest, not silent"
            )
        if c_real and w_info and w_info.dtype_dependent_unsupported:
            warnings.append(
                f"WGPU '{op}' has dtype/layout-dependent unsupported branches "
                f"(line {w_info.start_line}) — classify in manifest, not silent"
            )

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
        v_ok = usable(v_info) if v_info else False
        # device methods often lack run_* tokens — presence without immediate fail is ok
        if v_info is not None and not v_info.immediately_unsupported:
            v_ok = True
        w_ok = usable(w_info) if w_info else False
        if w_info is not None and not w_info.immediately_unsupported:
            w_ok = True
        if op in critical_device and c_real:
            if not v_ok:
                failures.append(f"CUDA device op '{op}' missing usable Vulkan impl")
            if not w_ok:
                failures.append(f"CUDA device op '{op}' missing usable WGPU impl")

    present_v = sum(1 for r in rows if r["vulkan"] == "yes")
    present_w = sum(1 for r in rows if r["wgpu"] == "yes")
    miss_v = sum(1 for r in rows if r["vulkan"] != "yes")
    miss_w = sum(1 for r in rows if r["wgpu"] != "yes")
    cuda_req = sum(1 for r in rows if r["cuda"] == "yes")

    manifest_counts = None
    mpath = manifest_path or (root / "docs" / "backend-parity-manifest.json")
    test_index = list_rust_test_functions(root)
    file_claim_count: Dict[str, int] = {}

    if not mpath.is_file():
        failures.append(f"required parity manifest missing: {mpath}")
    else:
        try:
            raw = json.loads(mpath.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "ops" in raw:
                data = raw["ops"]
                schema_version = raw.get("schema_version", 1)
            else:
                data = raw
                schema_version = 1
            if schema_version < 2:
                failures.append(f"parity manifest schema_version={schema_version} < 2")

            def count_status(key: str) -> Dict[str, int]:
                out: Dict[str, int] = {}
                for item in data:
                    st = item.get(key, "unknown")
                    out[st] = out.get(st, 0) + 1
                return out

            for item in data:
                op_name = item.get("op", "<unknown>")
                for field in PROFILE_FIELDS:
                    if field not in item:
                        if schema_version >= 2:
                            failures.append(
                                f"manifest op '{op_name}' missing profile field '{field}'"
                            )
                        continue
                    st = item[field]
                    if st not in ALLOWED_STATUSES:
                        failures.append(
                            f"manifest op '{op_name}' field '{field}' unknown status {st!r}"
                        )
                    if st == "Missing" and op_name in {
                        r["op"] for r in rows if r["cuda"] == "yes"
                    }:
                        failures.append(
                            f"manifest op '{op_name}' has Missing on {field}"
                        )
                    if st == "Verified":
                        validate_test_refs(
                            item, test_index, root, failures, file_claim_count
                        )
                        # dtype×layout evidence fields required for Verified compute ops
                        if (
                            not str(op_name).startswith("edge::")
                            and op_name
                            not in (
                                "cpu_fallback_policy",
                                "dtype",
                                "device",
                            )
                            and item.get("trait")
                            not in ("policy", "edge", "CudaSpecific")
                        ):
                            if not item.get("verified_dtypes") and not item.get(
                                "storage_dtypes"
                            ):
                                failures.append(
                                    f"manifest op '{op_name}' Verified without "
                                    "verified_dtypes/storage_dtypes evidence"
                                )
                            if item.get("contiguous") is None and item.get(
                                "strided"
                            ) is None:
                                failures.append(
                                    f"manifest op '{op_name}' Verified without "
                                    "layout flags (contiguous/strided)"
                                )
                    if field == "portable_webgpu_status" and st == "Verified":
                        portable_tests = item.get("portable_tests") or []
                        tests = item.get("tests") or []
                        banned = (
                            "backend_smoke_tests",
                            "gpu_parity_matrix",
                            "gpu_property_tests",
                            "gpu_metamorphic_tests",
                        )

                        def is_portable_evidence(path: object) -> bool:
                            s = str(path).lower()
                            if any(b in s for b in banned):
                                return False
                            return any(
                                t in s
                                for t in (
                                    "wasm",
                                    "portable",
                                    "browser",
                                    "candle-wasm",
                                )
                            )

                        if op_name not in ("cpu_fallback_policy",) and not (
                            any(is_portable_evidence(t) for t in portable_tests)
                            or any(is_portable_evidence(t) for t in tests)
                        ):
                            failures.append(
                                f"manifest op '{op_name}' portable Verified lacks "
                                "portable/WASM/browser evidence"
                            )
                    if st == "Optimized" and not item.get("bench"):
                        failures.append(
                            f"manifest op '{op_name}' Optimized on {field} but bench=false"
                        )

            for bare, n in file_claim_count.items():
                if n > MAX_GENERIC_FILE_SHARE:
                    failures.append(
                        f"generic test file '{bare}' cited as bare Verified evidence "
                        f"for {n} ops (max {MAX_GENERIC_FILE_SHARE}); use path::function"
                    )

            manifest_ops = {item.get("op") for item in data}
            for r in rows:
                if r["cuda"] == "yes" and r["op"] not in manifest_ops:
                    failures.append(
                        f"CUDA-required op '{r['op']}' missing from parity manifest"
                    )
            for dop in critical_device:
                if dop not in manifest_ops:
                    failures.append(
                        f"critical device op '{dop}' missing from parity manifest"
                    )

            manifest_counts = {
                "schema_version": schema_version,
                "vulkan": count_status("vulkan_status"),
                "native_webgpu": count_status("native_webgpu_status")
                if data and data and "native_webgpu_status" in data[0]
                else {},
                "portable_webgpu": count_status("portable_webgpu_status")
                if data and "portable_webgpu_status" in data[0]
                else {},
                "rows": len(data),
            }
        except Exception as e:  # noqa: BLE001
            failures.append(f"manifest load error: {e}")

    if as_json:
        print(
            json.dumps(
                {
                    "cuda_required_ops": cuda_req,
                    "rows": rows,
                    "failures": failures,
                    "warnings": warnings,
                    "static_presence": {
                        "vulkan_present": present_v,
                        "wgpu_present": present_w,
                        "vulkan_not_present": miss_v,
                        "wgpu_not_present": miss_w,
                    },
                    "manifest_counts": manifest_counts,
                },
                indent=2,
            )
        )
    else:
        print("=== Candle backend parity audit (strict) ===")
        print(f"repo: {root}")
        print(f"BackendStorage ops in trait: {len(storage_ops)}")
        print(f"CUDA-required (real impl):   {cuda_req}")
        print()
        print(f"{'op':28} {'cuda':12} {'vulkan':22} {'wgpu':22}")
        print("-" * 90)
        for r in rows:
            print(f"{r['op']:28} {r['cuda']:12} {r['vulkan']:22} {r['wgpu']:22}")
        print()
        print(f"  Vulkan present: {present_v}  not-present: {miss_v}")
        print(f"  WGPU   present: {present_w}  not-present: {miss_w}")
        if manifest_counts:
            print(f"Curated manifest rows: {manifest_counts['rows']}")
            print(f"  Vulkan:          {manifest_counts.get('vulkan')}")
            print(f"  Native WebGPU:   {manifest_counts.get('native_webgpu')}")
            print(f"  Portable WebGPU: {manifest_counts.get('portable_webgpu')}")
        if warnings:
            print(f"\nWARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"  - {w}")
        if failures:
            print(f"\nFAILURES ({len(failures)}):")
            for f in failures:
                print(f"  - {f}")
        else:
            print(
                "\nOK: each CUDA-required op has usable Vulkan AND Native WebGPU impl; "
                "manifest Verified entries carry function-level evidence."
            )
    return 1 if failures else 0


def run_self_tests() -> int:
    """Negative self-tests for audit logic (no GPU required)."""
    errors: List[str] = []

    # 1) immediate unsupported detection
    body_fail = 'fn foo(&self) -> Result<Self> {\n    return Err(unsupported("x"));\n}'
    if not is_immediately_unsupported(body_fail):
        errors.append("expected immediate unsupported for bare unsupported()")

    body_ok = (
        "fn foo(&self, layout: &Layout) -> Result<Self> {\n"
        "    self.run_unary_generic(layout, spirv)\n"
        "}"
    )
    if is_immediately_unsupported(body_ok):
        errors.append("did not expect immediate unsupported for run_unary path")

    # 2) usable() rejects presence-only without GPU work
    info = analyze_method("affine", 1, 'fn affine() { let x = 1; Err(unsupported("a")) }')
    # may be immediate
    if usable(info):
        errors.append("usable should be false for unsupported-only body")

    info2 = analyze_method(
        "affine",
        1,
        "fn affine(&self, l: &Layout, a: f64, b: f64) -> Result<Self> {\n"
        "    self.run_unary_generic_with_params(l, spirv, a, b)\n}",
    )
    if not usable(info2):
        errors.append("usable should be true for run_unary GPU path")

    # 3) per-backend failure rule (unit-level simulation)
    # Ensure both missing generates two failures when wired — covered by integration
    # on a temp tree with fake backends is heavy; check helper logic only.

    # 4) test ref parser
    f, fn = parse_test_ref("candle-core/tests/gpu_parity_matrix_tests.rs::parity_matmul_bf16")
    if fn != "parity_matmul_bf16" or "gpu_parity" not in f:
        errors.append(f"parse_test_ref failed: {f} {fn}")

    # 5) generic file share limit constant
    if MAX_GENERIC_FILE_SHARE < 1:
        errors.append("MAX_GENERIC_FILE_SHARE invalid")

    if errors:
        print("SELF-TEST FAILURES:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("SELF-TEST OK: audit helpers pass negative/positive unit checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run negative self-tests for audit helpers and exit",
    )
    args = ap.parse_args()
    if args.self_test:
        return run_self_tests()
    root = (args.repo or repo_root_from_script()).resolve()
    return run_audit(root, args.manifest, args.json)


if __name__ == "__main__":
    sys.exit(main())
