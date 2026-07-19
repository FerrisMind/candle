# Test summary (verification stage)

## gpu_parity_matrix_tests
- command: see logs/COMMANDS.txt
- result: 2 passed, 0 failed, 0 ignored, 0 filtered
- exit: 0
- env: CANDLE_REQUIRE_*=1, CANDLE_EXPECTED_GPU_NAME=RTX 3060, CANDLE_STRICT_NO_CPU_FALLBACK=1
- log: logs/gpu_parity_matrix_tests.log

## gpu_numerical_diff_tests
- result: 2 passed, 0 failed
- exit: 0
- cases: 22 per backend (primarily F32; F16 matmul optional; BF16 matmul skipped on this CUDA stack)
- report: numerical/diff_report.json
- log: logs/gpu_numerical_diff_tests.log

## fallback_runtime_audit
- result: ALL PASS (0 CPU fallbacks vulkan+wgpu)
- exit: 0
- log: fallback/fallback_runtime_audit.log

## backend_parity_microbench --suite
- raw: bench/microbench_raw.csv
- filtered: bench/microbench.csv
- slo: bench/slo_report.json
- exit: 0

## Not executed (documented gaps)
- LLM prefill/decode tokens/s, TTFT, VRAM
- full transformer e2e
- RMSNorm/LayerNorm/RoPE/attention fused benches
- browser WebGPU portable Verified
