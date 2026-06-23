#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    cargo_package: str
    cargo_example: str
    args: List[str]
    extra_env: Dict[str, str]
    kind: str
    bench_phase: Optional[str] = None  # "prefill" | "decode"
    backends: Optional[FrozenSet[str]] = None  # None = all backends


RUST_DUR_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ns|µs|us|ms|s|m|h)")
LLM_PROMPT_RE = re.compile(
    r"(?P<n>\d+)\s+prompt tokens processed:\s+(?P<tps>\d+(?:\.\d+)?)\s+token/s", re.I
)
LLM_GEN_RE = re.compile(
    r"(?P<n>\d+)\s+tokens generated:\s+(?P<tps>\d+(?:\.\d+)?)\s+token/s", re.I
)
PREFILL_NAME_RE = re.compile(r"__pp(\d+)$")
BERT_LOADED_RE = re.compile(r"Loaded and encoded\s+(?P<dur>.+)$")
BERT_TOOK_RE = re.compile(r"^Took\s+(?P<dur>.+)$")

LLM_KINDS = frozenset({"llama", "qwen", "quantized_qwen"})
PREFILL_SIZES = (512, 1024, 2048, 4096)
DECODE_SIZES = (128, 256)
DECODE_PROMPT_TOKENS = 16

# % of CUDA — Candle Vulkan/wgpu targets (RTX 3090/4090 class vs CUDA reference).
PERF_TARGETS: Dict[str, Dict[str, Dict[str, float]]] = {
    "vulkan": {
        "prefill": {"ok": 75.0, "good": 85.0, "excellent": 90.0},
        "decode": {"ok": 85.0, "good": 95.0, "excellent": 100.0},
    },
    "wgpu": {
        "prefill": {"ok": 60.0, "good": 75.0, "excellent": 85.0},
        "decode": {"ok": 80.0, "good": 90.0, "excellent": 100.0},
    },
}

# Cargo / example lines worth echoing while a run is in progress.
INTERESTING_LINE_RE = re.compile(
    r"(Compiling|Finished|Running|error\[|error:|warning:|panicked at|"
    r"tokens generated|prompt tokens processed|token/s|Took |Loaded and encoded|tokens/s|"
    r"Running on CPU|CANDLE_DEVICE|loading the model|model built|"
    r"starting the inference|TIMEOUT|avx:|loaded the model)",
    re.I,
)

# cuda first — speed reference for wgpu/vulkan comparisons.
DEFAULT_BACKENDS = ("cuda", "wgpu", "vulkan")
SUPPORTED_BACKENDS = frozenset(DEFAULT_BACKENDS)


def cuda_runtime_ok() -> Tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not found"
    proc = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, detail or f"nvidia-smi exit {proc.returncode}"
    if not proc.stdout.strip():
        return False, "no NVIDIA GPU reported by nvidia-smi"
    return True, ""


def normalize_backends(raw: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        backend = item.strip().lower()
        if not backend or backend in seen:
            continue
        if backend not in SUPPORTED_BACKENDS:
            supported = ", ".join(DEFAULT_BACKENDS)
            raise SystemExit(f"unsupported backend: {backend} (use: {supported})")
        out.append(backend)
        seen.add(backend)
    return out


def spec_prefill_tokens(name: str) -> Optional[int]:
    m = PREFILL_NAME_RE.search(name)
    return int(m.group(1)) if m else None


def skip_spec_reason(
    spec: ExampleSpec,
    backend: str,
    image: str,
    max_prefill_tokens: Optional[int],
) -> Optional[str]:
    if spec.backends is not None and backend not in spec.backends:
        allowed = ", ".join(sorted(spec.backends))
        return f"runs on {allowed} only (dense safetensors — use gguf on {backend})"
    if spec.bench_phase == "prefill" and max_prefill_tokens is not None:
        n = spec_prefill_tokens(spec.name)
        if n is not None and n > max_prefill_tokens:
            return f"prefill {n} > --max-prefill-tokens {max_prefill_tokens}"
    if "__IMAGE__" in spec.args:
        image_path = Path(image).expanduser() if image else None
        if not image or image_path is None or not image_path.is_file():
            return f"image not found: {image or '(empty)'}"
    return None


def cuda_reference_tps(
    rows: List[Dict[str, Any]], example: str, phase: Optional[str] = None
) -> Optional[float]:
    for row in rows:
        if row.get("backend") != "cuda" or row.get("name") != example:
            continue
        if row.get("status") != "ok":
            continue
        avg = row.get("avg") or {}
        if phase == "prefill":
            tps = avg.get("prompt_tokens_per_s")
        elif phase == "decode":
            tps = avg.get("tokens_per_s")
        else:
            tps = avg.get("tokens_per_s") or avg.get("prompt_tokens_per_s")
        if tps is not None:
            return float(tps)
    return None


def primary_tps(avg: Dict[str, Any], phase: Optional[str]) -> Optional[float]:
    if phase == "prefill":
        return avg.get("prompt_tokens_per_s")
    if phase == "decode":
        return avg.get("tokens_per_s")
    return avg.get("tokens_per_s") or avg.get("prompt_tokens_per_s")


def grade_perf(backend: str, phase: Optional[str], pct: Optional[float]) -> str:
    if pct is None or phase not in ("prefill", "decode"):
        return "-"
    targets = PERF_TARGETS.get(backend, {}).get(phase)
    if not targets:
        return "-"
    if pct >= targets["excellent"]:
        return "excellent"
    if pct >= targets["good"]:
        return "good"
    if pct >= targets["ok"]:
        return "ok"
    return "below"


def parse_llm_metrics(out: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    prompt_m = None
    gen_m = None
    for line in out.splitlines():
        pm = LLM_PROMPT_RE.search(line)
        if pm:
            prompt_m = pm
        gm = LLM_GEN_RE.search(line)
        if gm:
            gen_m = gm
    if prompt_m:
        n = int(prompt_m.group("n"))
        tps = float(prompt_m.group("tps"))
        metrics["prompt_tokens"] = n
        metrics["prompt_tokens_per_s"] = tps
        metrics["prompt_s"] = (n / tps) if tps > 0 else None
    if gen_m:
        n = int(gen_m.group("n"))
        tps = float(gen_m.group("tps"))
        metrics["generated_tokens"] = n
        metrics["tokens_per_s"] = tps
        metrics["infer_s"] = (n / tps) if tps > 0 else None
    return metrics


def ts() -> str:
    return time.strftime("%H:%M:%S")


def parse_rust_debug_duration(s: str) -> Optional[float]:
    s = s.strip().strip(",")
    if not s:
        return None
    total = 0.0
    matched_any = False
    for m in RUST_DUR_RE.finditer(s):
        matched_any = True
        value = float(m.group("value"))
        unit = m.group("unit")
        if unit == "ns":
            total += value * 1e-9
        elif unit in ("us", "µs"):
            total += value * 1e-6
        elif unit == "ms":
            total += value * 1e-3
        elif unit == "s":
            total += value
        elif unit == "m":
            total += value * 60.0
        elif unit == "h":
            total += value * 3600.0
    return total if matched_any else None


def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def fmt_s(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def fmt_tps(tps: Optional[float]) -> str:
    if tps is None:
        return "-"
    return f"{tps:.2f} tok/s"


def extract_failure_reason(out: str) -> Optional[str]:
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("error:") or s.startswith("Error:") or "panicked at" in s:
            return s[:140]
        if "invalid value" in s and "for '--" in s:
            return s[:140]
    return None


class Reporter:
    def __init__(self, total_jobs: int, heartbeat_s: float, stream_output: bool) -> None:
        self.total_jobs = total_jobs
        self.heartbeat_s = heartbeat_s
        self.stream_output = stream_output
        self.job_idx = 0
        self._run_start: float = 0.0
        self._global_start = time.perf_counter()
        self.rows: List[Dict[str, Any]] = []

    def _log(self, msg: str) -> None:
        print(f"[{ts()}] {msg}", flush=True)

    def banner(self, backends: List[str], runs: int, models_root: Path) -> None:
        self._log("=" * 72)
        self._log("candle bench_examples — live progress enabled")
        self._log(f"backends: {', '.join(backends)} | runs/example: {runs} | jobs: {self.total_jobs}")
        if "cuda" in backends:
            self._log("cuda runs first as the speed reference for other GPU backends")
        self._log(f"models: {models_root}")
        self._log("=" * 72)

    def skip_example(self, backend: str, name: str, reason: str) -> None:
        self.job_idx += 1
        self._log(f"[{self.job_idx}/{self.total_jobs}] SKIP {backend} :: {name} — {reason}")
        self.rows.append(
            {"backend": backend, "name": name, "status": "skipped", "reason": reason}
        )
        self._print_running_table()

    def begin_run(self, backend: str, name: str, run_idx: int, runs: int, cmd: str) -> None:
        if run_idx == 0:
            self.job_idx += 1
            self._log("-" * 72)
            self._log(f"[{self.job_idx}/{self.total_jobs}] START {backend} :: {name}")
            self._log(f"cmd: {cmd}")
        self._run_start = time.perf_counter()
        self._log(f"  run {run_idx + 1}/{runs} …")

    def on_output_line(self, line: str) -> None:
        if not self.stream_output or not line:
            return
        if INTERESTING_LINE_RE.search(line):
            self._log(f"  | {line}")

    def heartbeat(self, backend: str, name: str, run_idx: int, runs: int) -> None:
        elapsed = time.perf_counter() - self._run_start
        total = time.perf_counter() - self._global_start
        self._log(
            f"  … still running {backend} :: {name} "
            f"(run {run_idx + 1}/{runs}, {elapsed:.0f}s this run, {total:.0f}s total)"
        )

    def end_run(
        self,
        backend: str,
        name: str,
        run_idx: int,
        runs: int,
        code: int,
        wall_s: float,
        metrics: Optional[Dict[str, Any]],
        log_path: Path,
        error: Optional[str],
    ) -> None:
        status = "OK" if code == 0 else "FAIL"
        parts = [f"  run {run_idx + 1}/{runs} {status} in {fmt_s(wall_s)}"]
        if metrics:
            if metrics.get("prompt_tokens_per_s") is not None:
                parts.append(f"prompt={fmt_tps(metrics.get('prompt_tokens_per_s'))}")
            if metrics.get("tokens_per_s") is not None:
                parts.append(f"infer={fmt_tps(metrics.get('tokens_per_s'))}")
            if metrics.get("infer_s") is not None:
                parts.append(f"gen={fmt_s(metrics.get('infer_s'))}")
            if metrics.get("prompt_s") is not None:
                parts.append(f"prompt/load={fmt_s(metrics.get('prompt_s'))}")
        if error:
            parts.append(error)
        parts.append(f"log={log_path}")
        self._log(" | ".join(parts))

    def end_example(self, spec_entry: Dict[str, Any]) -> None:
        avg = spec_entry.get("avg", {})
        status = spec_entry.get("status", "?")
        name = spec_entry["name"]
        backend = spec_entry["backend"]
        vs_cuda = avg.get("vs_cuda_pct")
        vs_s = f"{vs_cuda:.0f}%" if vs_cuda is not None else "-"
        self._log(
            f"DONE {backend} :: {name} — {status} | "
            f"ok_runs={avg.get('successful_runs', 0)} | "
            f"phase={spec_entry.get('bench_phase') or '-'} | "
            f"prompt tok/s={fmt_tps(avg.get('prompt_tokens_per_s'))} | "
            f"decode tok/s={fmt_tps(avg.get('tokens_per_s'))} | "
            f"vs cuda={vs_s}"
        )
        if avg.get("perf_grade"):
            self._log(f"  grade: {avg['perf_grade']}")
        row: Dict[str, Any] = {
            "backend": backend,
            "name": name,
            "status": status,
            "avg": avg,
        }
        if spec_entry.get("bench_phase"):
            row["bench_phase"] = spec_entry["bench_phase"]
        if spec_entry.get("error_detail"):
            row["error_detail"] = spec_entry["error_detail"]
        if spec_entry.get("skip_reason"):
            row["skip_reason"] = spec_entry["skip_reason"]
        self.rows.append(row)
        self._print_running_table()

    def _print_running_table(self) -> None:
        if not self.rows:
            return
        self._log("— current summary —")
        header = (
            f"{'backend':8} {'example':28} {'phase':7} {'status':8} "
            f"{'tok/s':10} {'vs cuda':8} {'grade':9}  note"
        )
        self._log(header)
        for row in self.rows:
            avg = row.get("avg") or {}
            phase = row.get("bench_phase") or "-"
            note = row.get("error_detail") or row.get("skip_reason") or row.get("reason") or ""
            if len(note) > 36:
                note = note[:33] + "..."
            tps = primary_tps(avg, row.get("bench_phase"))
            ref = cuda_reference_tps(self.rows, row["name"], row.get("bench_phase"))
            if ref is not None and tps is not None and row.get("backend") != "cuda":
                vs = f"{100.0 * float(tps) / ref:.0f}%"
                grade = grade_perf(row["backend"], row.get("bench_phase"), 100.0 * float(tps) / ref)
            elif row.get("backend") == "cuda" and tps is not None:
                vs = "ref"
                grade = "-"
            else:
                vs = "-"
                grade = "-"
            self._log(
                f"{row['backend']:8} {row['name']:28} {phase:7} {row['status']:8} "
                f"{fmt_tps(tps):>10} {vs:>8} {grade:9}  {note}"
            )
        self._log("—" * 36)

    def final_summary(self, out_path: Path, log_dir: Path) -> None:
        total = time.perf_counter() - self._global_start
        self._log("=" * 72)
        self._log(f"FINISHED in {total:.1f}s")
        self._log(f"results: {out_path}")
        self._log(f"logs:    {log_dir}")
        self._print_running_table()
        self._log("=" * 72)


def run_once_live(
    cmd: List[str],
    env: Dict[str, str],
    cwd: Path,
    timeout_s: Optional[int],
    reporter: Reporter,
    backend: str,
    name: str,
    run_idx: int,
    runs: int,
) -> Tuple[int, float, str]:
    start = time.perf_counter()
    lines: List[str] = []
    stop = threading.Event()

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            reporter.on_output_line(line.rstrip("\n"))

    def heartbeat() -> None:
        while not stop.wait(reporter.heartbeat_s):
            if proc.poll() is None:
                reporter.heartbeat(backend, name, run_idx, runs)

    t_reader = threading.Thread(target=reader, daemon=True)
    t_hb = threading.Thread(target=heartbeat, daemon=True)
    t_reader.start()
    t_hb.start()

    code = 1
    try:
        code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        lines.append(f"\n[bench_examples] TIMEOUT after {timeout_s}s\n")
        code = 124
    finally:
        stop.set()
        t_reader.join(timeout=2.0)

    wall_s = time.perf_counter() - start
    return code, wall_s, "".join(lines)


def parse_metrics(kind: str, wall_s: float, out: str, bench_phase: Optional[str] = None) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"wall_s": wall_s}

    if kind in LLM_KINDS:
        metrics.update(parse_llm_metrics(out))
        return metrics

    if kind == "bert":
        loaded_s: Optional[float] = None
        took_s: List[float] = []
        for line in out.splitlines():
            m1 = BERT_LOADED_RE.search(line)
            if m1:
                loaded_s = parse_rust_debug_duration(m1.group("dur"))
            m2 = BERT_TOOK_RE.search(line.strip())
            if m2:
                v = parse_rust_debug_duration(m2.group("dur"))
                if v is not None:
                    took_s.append(v)
        metrics["prompt_s"] = loaded_s
        metrics["infer_s_runs"] = took_s
        metrics["infer_s"] = mean(took_s)
        return metrics

    return metrics


def cargo_run_cmd(pkg: str, features: str, example: str, args: List[str]) -> List[str]:
    cmd = ["cargo", "run", "-p", pkg, "--release", "--features", features, "--example", example, "--"]
    cmd.extend(args)
    return cmd


def append_llm_matrix(
    specs: List[ExampleSpec],
    base_name: str,
    cargo_example: str,
    kind: str,
    common_args: List[str],
    extra_env: Dict[str, str],
    prefill_sizes: Tuple[int, ...],
    backends: Optional[FrozenSet[str]] = None,
) -> None:
    bench_common = common_args + ["--temperature", "0", "--repeat-penalty", "1"]
    for n in prefill_sizes:
        specs.append(
            ExampleSpec(
                name=f"{base_name}__pp{n}",
                cargo_package="candle-examples",
                cargo_example=cargo_example,
                args=bench_common
                + ["--bench-prompt-tokens", str(n), "--sample-len", "1"],
                extra_env=extra_env,
                kind=kind,
                bench_phase="prefill",
                backends=backends,
            )
        )
    for n in DECODE_SIZES:
        specs.append(
            ExampleSpec(
                name=f"{base_name}__tg{n}",
                cargo_package="candle-examples",
                cargo_example=cargo_example,
                args=bench_common
                + [
                    "--bench-prompt-tokens",
                    str(DECODE_PROMPT_TOKENS),
                    "--sample-len",
                    str(n + 1),
                ],
                extra_env=extra_env,
                kind=kind,
                bench_phase="decode",
                backends=backends,
            )
        )


def build_specs(
    models_root: Path, llm_matrix: bool, prefill_sizes: Tuple[int, ...]
) -> List[ExampleSpec]:
    specs: List[ExampleSpec] = []

    # Smaller examples first — drivers can lag reclaiming VRAM after OOM/crashes.
    bge_dir = models_root / "bge-small-en-v1.5"
    if bge_dir.is_dir():
        specs.append(
            ExampleSpec(
                name="bge-small-en-v1.5",
                cargo_package="candle-examples",
                cargo_example="bert",
                args=["--prompt", "hello world", "--n", "3", "--weight-path", str(bge_dir)],
                extra_env={"RUST_BACKTRACE": "0"},
                kind="bert",
            )
        )

    minilm_dir = models_root / "all-MiniLM-L6-v2"
    minilm_args = ["--prompt", "hello world", "--n", "3"]
    if minilm_dir.is_dir():
        minilm_args += ["--weight-path", str(minilm_dir)]
    specs.append(
        ExampleSpec(
            name="all-MiniLM-L6-v2",
            cargo_package="candle-examples",
            cargo_example="bert",
            args=minilm_args,
            extra_env={"RUST_BACKTRACE": "0"},
            kind="bert",
        )
    )

    t5_dir = models_root / "t5-small"
    if (t5_dir / "config.json").is_file():
        specs.append(
            ExampleSpec(
                name="t5-small",
                cargo_package="candle-examples",
                cargo_example="t5",
                args=[
                    "--which",
                    "t5-small",
                    "--decode",
                    "--prompt",
                    "translate English to German: I like pizza.",
                    "--config-file",
                    str(t5_dir / "config.json"),
                    "--model-file",
                    str(t5_dir / "model.safetensors"),
                    "--tokenizer-file",
                    str(t5_dir / "tokenizer.json"),
                ],
                extra_env={},
                kind="generic",
            )
        )

    specs.append(
        ExampleSpec(
            name="resnet-50",
            cargo_package="candle-examples",
            cargo_example="resnet",
            args=["--which", "50", "--image", "__IMAGE__"],
            extra_env={},
            kind="generic",
        )
    )

    gguf_dir = models_root / "Qwen3-0.6B-GGUF"
    gguf_path = gguf_dir / "Qwen3-0.6B-Q4_K_M.gguf"
    if llm_matrix and gguf_path.is_file():
        gguf_args = ["--model", str(gguf_path)]
        tokenizer_path = gguf_dir / "tokenizer.json"
        if tokenizer_path.is_file():
            gguf_args += ["--tokenizer", str(tokenizer_path)]
        append_llm_matrix(
            specs,
            "qwen3-0.6b-gguf",
            "quantized-qwen3",
            "quantized_qwen",
            gguf_args,
            {},
            prefill_sizes,
        )

    qwen_dir = models_root / "Qwen3-0.6B"
    if llm_matrix and qwen_dir.is_dir():
        qwen_args = [
            "--model",
            "3-0.6b",
            "--dtype",
            "bf16",
            "--no-chat-template",
            "--weight-path",
            str(qwen_dir),
        ]
        append_llm_matrix(
            specs,
            "qwen3-0.6b",
            "qwen",
            "qwen",
            qwen_args,
            {},
            prefill_sizes,
        )

    llama_dir = models_root / "Llama-3.2-1B-Instruct"
    if llm_matrix and llama_dir.is_dir():
        llama_args = [
            "--which",
            "v32-1b-instruct",
            "--dtype",
            "bf16",
            "--weight-path",
            str(llama_dir),
        ]
        append_llm_matrix(
            specs,
            "llama-3.2-1b-instruct",
            "llama",
            "llama",
            llama_args,
            {},
            prefill_sizes,
        )

    return specs


def count_jobs(
    specs: List[ExampleSpec],
    backends: List[str],
    image: str,
    max_prefill_tokens: Optional[int],
) -> int:
    n = 0
    cuda_ok, _ = cuda_runtime_ok()
    for backend in backends:
        if backend == "cuda" and not cuda_ok:
            continue
        for spec in specs:
            if skip_spec_reason(spec, backend, image, max_prefill_tokens):
                continue
            n += 1
    return n


def parse_prefill_sizes(raw: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise SystemExit(f"invalid prefill size: {n}")
        out.append(n)
    if not out:
        raise SystemExit("empty --prefill-sizes")
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run Candle examples on cuda (reference), wgpu, and vulkan "
            "with live progress and averaged metrics."
        )
    )
    ap.add_argument(
        "--backend",
        action="append",
        default=[],
        help="cuda, wgpu, and/or vulkan (default: cuda, wgpu, vulkan — cuda first as speed baseline)",
    )
    ap.add_argument(
        "--functional-only",
        action="store_true",
        help="Skip LLM prefill/decode matrix (bert/t5/resnet only)",
    )
    ap.add_argument("--models-root", default="/home/mod479711/Downloads/models")
    ap.add_argument(
        "--prefill-sizes",
        default=",".join(str(n) for n in PREFILL_SIZES),
        help="Comma-separated prefill token counts (default: 512,1024,2048,4096)",
    )
    ap.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=None,
        help="Skip prefill specs above this size (e.g. 2048 on 12GB GPUs)",
    )
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument(
        "--image",
        default="/home/mod479711/Downloads/360_F_597560812_N83lzoaNTh4DsqzPVZUpfJJMA2IOIJYe.jpg",
        help="Image for resnet-50",
    )
    ap.add_argument("--json-out", default="bench_results.json")
    ap.add_argument("--log-dir", default="bench_logs")
    ap.add_argument(
        "--heartbeat-s",
        type=float,
        default=15.0,
        help="Print alive status every N seconds while a run is active",
    )
    ap.add_argument(
        "--no-stream-output",
        action="store_true",
        help="Do not echo interesting cargo/example lines during runs",
    )
    args = ap.parse_args()

    try:
        backends = normalize_backends(args.backend) if args.backend else list(DEFAULT_BACKENDS)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    cuda_ok, cuda_skip_reason = cuda_runtime_ok()
    if "cuda" in backends and not cuda_ok:
        print(f"warning: skipping cuda backend — {cuda_skip_reason}", file=sys.stderr)
        backends = [b for b in backends if b != "cuda"]
        if not backends:
            print("no backends left to run", file=sys.stderr)
            return 2

    repo_root = Path(__file__).resolve().parent
    models_root = Path(args.models_root).expanduser().resolve()
    log_dir = (repo_root / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        prefill_sizes = parse_prefill_sizes(args.prefill_sizes)
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2

    specs = build_specs(models_root, llm_matrix=not args.functional_only, prefill_sizes=prefill_sizes)
    reporter = Reporter(
        total_jobs=count_jobs(specs, backends, args.image, args.max_prefill_tokens),
        heartbeat_s=args.heartbeat_s,
        stream_output=not args.no_stream_output,
    )
    reporter.banner(backends, args.runs, models_root)
    if not specs:
        reporter._log("no local model specs found — check --models-root")
        return 1
    llm_specs = [s.name for s in specs if s.bench_phase]
    func_specs = [s.name for s in specs if not s.bench_phase]
    if llm_specs:
        reporter._log(
            f"LLM matrix ({len(llm_specs)}): pp{list(prefill_sizes)} + "
            f"tg{list(DECODE_SIZES)} vs CUDA"
        )
    if func_specs:
        reporter._log(f"functional ({len(func_specs)}): {', '.join(func_specs)}")

    results: Dict[str, Any] = {
        "repo_root": str(repo_root),
        "models_root": str(models_root),
        "backends": backends,
        "reference_backend": "cuda" if "cuda" in backends else None,
        "perf_targets_pct_of_cuda": PERF_TARGETS,
        "llm_matrix": {
            "prefill_sizes": list(prefill_sizes),
            "decode_sizes": list(DECODE_SIZES),
            "decode_prompt_tokens": DECODE_PROMPT_TOKENS,
        },
        "runs": args.runs,
        "examples": [],
    }

    base_env = os.environ.copy()
    base_env.setdefault("RUST_BACKTRACE", "1")

    for backend in backends:
        for spec in specs:
            skip_reason = skip_spec_reason(spec, backend, args.image, args.max_prefill_tokens)
            if skip_reason:
                entry = {
                    "name": spec.name,
                    "backend": backend,
                    "status": "skipped",
                    "skip_reason": skip_reason,
                    "runs": [],
                    "avg": {"successful_runs": 0},
                }
                if spec.bench_phase:
                    entry["bench_phase"] = spec.bench_phase
                results["examples"].append(entry)
                reporter.skip_example(backend, spec.name, skip_reason)
                continue

            spec_args = list(spec.args)
            if "__IMAGE__" in spec_args:
                image_path = Path(args.image).expanduser()
                spec_args = [
                    str(image_path) if a == "__IMAGE__" else a for a in spec_args
                ]

            cmd = cargo_run_cmd(spec.cargo_package, backend, spec.cargo_example, spec_args)
            cmd_s = " ".join(shlex.quote(c) for c in cmd)
            env = dict(base_env)
            env["CANDLE_DEVICE"] = backend
            env.update(spec.extra_env)

            spec_entry: Dict[str, Any] = {
                "name": spec.name,
                "backend": backend,
                "cmd": cmd_s,
                "runs": [],
                "status": "ok",
            }
            if spec.bench_phase:
                spec_entry["bench_phase"] = spec.bench_phase

            for run_idx in range(args.runs):
                reporter.begin_run(backend, spec.name, run_idx, args.runs, cmd_s)
                code, wall_s, out = run_once_live(
                    cmd,
                    env,
                    repo_root,
                    args.timeout_s,
                    reporter,
                    backend,
                    spec.name,
                    run_idx,
                    args.runs,
                )

                log_path = log_dir / f"{spec.name}__{backend}__run{run_idx}.log"
                log_path.write_text(out, encoding="utf-8", errors="replace")

                run_rec: Dict[str, Any] = {
                    "run_idx": run_idx,
                    "exit_code": code,
                    "wall_s": wall_s,
                    "log_path": str(log_path),
                }

                if code != 0:
                    err = "timeout" if code == 124 else f"exit {code}"
                    detail = extract_failure_reason(out)
                    if detail:
                        err = f"{err}: {detail}"
                        spec_entry["error_detail"] = detail
                    run_rec["error"] = err
                    spec_entry["runs"].append(run_rec)
                    spec_entry["status"] = "failed"
                    reporter.end_run(
                        backend,
                        spec.name,
                        run_idx,
                        args.runs,
                        code,
                        wall_s,
                        None,
                        log_path,
                        err,
                    )
                    reporter._log(
                        f"  stopping remaining runs for {backend} :: {spec.name} after failure"
                    )
                    break

                metrics = parse_metrics(spec.kind, wall_s, out, spec.bench_phase)
                run_rec["metrics"] = metrics
                spec_entry["runs"].append(run_rec)
                reporter.end_run(
                    backend,
                    spec.name,
                    run_idx,
                    args.runs,
                    code,
                    wall_s,
                    metrics,
                    log_path,
                    None,
                )

            ok_metrics = [
                r["metrics"]
                for r in spec_entry["runs"]
                if r.get("exit_code") == 0 and "metrics" in r
            ]
            prompt_s = [m.get("prompt_s") for m in ok_metrics if m.get("prompt_s") is not None]
            infer_s = [m.get("infer_s") for m in ok_metrics if m.get("infer_s") is not None]
            tps = [m.get("tokens_per_s") for m in ok_metrics if m.get("tokens_per_s") is not None]
            prompt_tps = [
                m.get("prompt_tokens_per_s")
                for m in ok_metrics
                if m.get("prompt_tokens_per_s") is not None
            ]
            spec_entry["avg"] = {
                "prompt_s": mean(prompt_s),
                "infer_s": mean(infer_s),
                "tokens_per_s": mean(tps),
                "prompt_tokens_per_s": mean(prompt_tps),
                "wall_s": mean([m.get("wall_s") for m in ok_metrics if m.get("wall_s") is not None]),
                "successful_runs": len(ok_metrics),
            }
            phase = spec.bench_phase
            ref_tps = cuda_reference_tps(reporter.rows, spec.name, phase)
            primary = primary_tps(spec_entry["avg"], phase)
            if ref_tps is not None and primary is not None and backend != "cuda":
                pct = 100.0 * float(primary) / ref_tps
                spec_entry["avg"]["vs_cuda_pct"] = pct
                spec_entry["avg"]["perf_grade"] = grade_perf(backend, phase, pct)

            results["examples"].append(spec_entry)
            reporter.end_example(spec_entry)

            out_path = (repo_root / args.json_out).resolve()
            out_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    out_path = (repo_root / args.json_out).resolve()
    reporter.final_summary(out_path, log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
