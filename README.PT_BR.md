<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-232323" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-3ABF7A" alt="Português"></a>
</p>

---

<p align="center">
  <b>Fork do Candle com backends nativos Vulkan e WGPU / WebGPU.</b><br>
  Foco em paridade com CUDA para inferência em Linux, Windows, macOS, Android e WASM.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache--2.0%20%2F%20MIT-2ea44f" alt="Apache 2.0 / MIT">
  <img src="https://img.shields.io/badge/Rust-edition%202021-93450a?logo=rust" alt="Rust edition 2021">
  <img src="https://img.shields.io/badge/Backends-Vulkan%20%2B%20WGPU-5B7CFA" alt="Vulkan and WGPU">
  <img src="https://img.shields.io/badge/Fork-0.0.174-d4730e" alt="Fork version 0.0.174">
  <img src="https://img.shields.io/badge/Upstream-Candle%200.11.0-232323" alt="Upstream Candle 0.11.0">
  <img src="https://img.shields.io/badge/Branch-wgpu%2Fvulkan-232323" alt="wgpu/vulkan branch">
</p>

<h1 align="center">Candle / wgpu · vulkan</h1>

## Matriz de plataformas

| Backend | Feature | Linux | Windows | macOS | Android | WASM |
|---------|---------|:-----:|:-------:|:-----:|:-------:|:----:|
| CPU | (default) | ✅ | ✅ | ✅ | ✅ | ✅ |
| CUDA | `cuda` | ✅ | ✅ | ❌ | ❌ | ❌ |
| Metal | `metal` | ❌ | ❌ | ✅ | ❌ | ❌ |
| WGPU | `wgpu` | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vulkan | `vulkan` | ✅ | ✅ | ❌ | ✅ | ❌ |

Os backends **CPU** e **CUDA** permanecem sincronizados diretamente com o [huggingface/candle](https://github.com/huggingface/candle) `main`. Nada desses backends upstream foi removido; este fork apenas adiciona Vulkan e WGPU por cima.

## Correspondência com o upstream

| Fork ([FerrisMind/candle](https://github.com/FerrisMind/candle)) | Upstream ([huggingface/candle](https://github.com/huggingface/candle)) |
|------------------------------------------------------------------|------------------------------------------------------------------------|
| Branch `wgpu/vulkan` | `main` |
| Crates do fork **0.0.174** (`candle-core` / `candle-nn` / `candle-transformers` / `candle-examples` / `candle-*-kernels` Vulkan e WGPU); inalterados ficam **0.11.0** | Candle **0.11.0** |
| Último sync CPU/CUDA [`b3e5b40f`](https://github.com/FerrisMind/candle/commit/b3e5b40f) (2026-08-17) | Tip [`162b59b9`](https://github.com/FerrisMind/candle/commit/162b59b9) (#3892) |
| Somente no fork | **Vulkan** + **WGPU / WebGPU** nativos |

CPU e CUDA ficam sincronizados com o `main` upstream (nada removido). Vulkan e WGPU existem apenas neste fork.

## Índice

- [Matriz de plataformas](#matriz-de-plataformas)
- [Correspondência com o upstream](#correspondência-com-o-upstream)
- [O que é isso?](#o-que-é-isso)
- [Principais recursos](#principais-recursos)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Início rápido](#início-rápido)
- [Desempenho dos backends](#desempenho-dos-backends)
- [Requisitos do sistema](#requisitos-do-sistema)
- [Licença](#licença)

## O que é isso?

Este é um fork de [huggingface/candle](https://github.com/huggingface/candle) na branch padrão [`wgpu/vulkan`](https://github.com/FerrisMind/candle/tree/wgpu/vulkan). O upstream continua sendo a fonte da API de tensores, dos modelos e dos caminhos CPU / CUDA / Metal — esses backends ficam sincronizados com `main` e não são removidos. Nesta branch foram adicionados e reforçados:

- um backend de compute **Vulkan nativo** (`ash` + SPIR-V)
- um backend de compute **WGPU / WebGPU** (`wgpu` + WGSL)
- documentação de paridade, smoke tests e cobertura diferencial em relação ao CUDA

A paridade é acompanhada em três perfis separados (não misture os resultados):

| Perfil | Significado |
|--------|-------------|
| Native Vulkan | Vulkan / SPIR-V direto |
| Native WebGPU | `wgpu` nativo com detecção de features em runtime |
| Portable WebGPU | WGSL seguro para browser / WASM; sem claims native-only |

Documentos normativos:

- [`docs/backend-parity-spec.md`](./docs/backend-parity-spec.md)
- [`docs/backend-parity.md`](./docs/backend-parity.md)
- [`docs/backend-parity-manifest.json`](./docs/backend-parity-manifest.json)

## Principais recursos

- As Cargo features `vulkan` e `wgpu` habilitam os novos backends de GPU.
- Seleção de dispositivo via `CANDLE_DEVICE=vulkan|wgpu` (também `cuda`, `metal`, `cpu`).
- Escolha de adapter: `CANDLE_VULKAN_DEVICE_NAME`, `CANDLE_WGPU_ADAPTER_NAME`.
- Override da API WGPU: `WGPU_BACKEND=vulkan|dx12|metal|gl`.
- Sem compute em CPU escondido como GPU; sem cast silencioso de dtype para F32.
- Audit estático de paridade, smoke tests, matriz diferencial CUDA, audit de fallback e harness de bench dos examples.

## Estrutura do repositório

| Caminho | Propósito |
|---------|-----------|
| `candle-core` | API de tensores, devices, storage e ops Vulkan / WGPU |
| `candle-vulkan-kernels` | Shaders de compute SPIR-V |
| `candle-wgpu-kernels` | Shaders de compute WGSL |
| `candle-nn` | Camadas (incl. MoE e dispatch de flash-attn) |
| `candle-transformers` | Implementações de modelos |
| `candle-examples` | Examples executáveis para cobertura e2e dos backends |
| `docs/` | Especificação de paridade, manifesto e evidências |
| `scripts/` | Audit de paridade e tooling relacionado |
| `bench_examples.py` | Harness multi-backend de throughput dos examples |

## Início rápido

### Rodar um example em Vulkan ou WGPU

```powershell
$env:CANDLE_DEVICE = "vulkan"   # ou "wgpu"
cargo run -p candle-examples --release --features vulkan --example quantized-qwen3 -- --model <path-to-gguf>
cargo run -p candle-examples --release --features wgpu --example quantized-qwen3 -- --model <path-to-gguf>
```

### Checagens de paridade

```powershell
python scripts/backend_parity_audit.py

cargo test -p candle-core --features vulkan --test backend_smoke_tests
cargo test -p candle-core --features wgpu --test backend_smoke_tests

# Matriz diferencial CUDA (requer GPUs)
$env:CANDLE_REQUIRE_CUDA_TEST_DEVICE = "1"
$env:CANDLE_REQUIRE_VULKAN_TEST_DEVICE = "1"
$env:CANDLE_REQUIRE_WGPU_TEST_DEVICE = "1"
cargo test -p candle-core --features "cuda,vulkan,wgpu" --test gpu_parity_matrix_tests

cargo run -p candle-core --release --features "vulkan,wgpu" --example fallback_runtime_audit
cargo run -p candle-core --release --features "cuda,vulkan,wgpu" --example backend_parity_microbench -- --suite

python bench_examples.py --models-root <models-root> --backend cuda --backend vulkan --backend wgpu
```

## Desempenho dos backends

Throughput end-to-end para [`quantized-qwen3`](./candle-examples/examples/quantized-qwen3/) (**Qwen3-0.6B-GGUF Q4_K_M**, release, baseline CUDA na mesma sessão).

Hardware: **RTX 3060 12 GB**, **Ryzen 7 3700X**, **64 GB DDR4**. `%CUDA` é o throughput relativo ao CUDA; Min / Normal / Goal são tiers de SLO de [`bench_examples.py`](./bench_examples.py).

| Backend | Phase | Cell | tok/s | %CUDA | Min | Normal | Goal | Verdict |
|---------|-------|------|------:|------:|----:|-------:|-----:|---------|
| Vulkan | decode | tg128 | 121.51 | 202% | 75 | 90 | 90+ | PASS ×3 |
| Vulkan | decode | tg256 | 121.26 | 203% | 75 | 90 | 90+ | PASS ×3 |
| Vulkan | prefill | pp512 | 120.72 | 205% | 85 | 95 | 95+ | PASS ×3 |
| Vulkan | prefill | pp1024 | 117.55 | 207% | 85 | 95 | 95+ | PASS ×3 |
| Vulkan | prefill | pp2048 | 112.03 | 190% | 85 | 95 | 95+ | PASS ×3 |
| Vulkan | prefill | pp4096 | 96.46 | 220% | 85 | 95 | 95+ | PASS ×3 |
| WGPU | decode | tg128 | 53.85 | 90% | 10 | 18 | 30+ | PASS ×3 |
| WGPU | decode | tg256 | 52.99 | 89% | 10 | 18 | 30+ | PASS ×3 |
| WGPU | prefill | pp512 | 52.87 | 90% | 12 | 20 | 35+ | PASS ×3 |
| WGPU | prefill | pp1024 | 45.18 | 80% | 12 | 20 | 35+ | PASS ×3 |
| WGPU | prefill | pp2048 | 33.50 | 57% | 12 | 20 | 35+ | PASS ×3 |
| WGPU | prefill | pp4096 | 22.01 | 50% | 12 | 22 | 38+ | PASS ×3 |

Baseline CUDA (mesma sessão): tg128 60.09, tg256 59.57, pp512 58.79, pp1024 56.74, pp2048 58.78, pp4096 43.91 tok/s.

Metas de SLO de release (end-to-end vs CUDA): **Vulkan ≤ 15% mais lento** (stretch 10%); **native WebGPU ≤ 30% mais lento** (stretch 20%). Portable WebGPU não tem % fixo em relação ao CUDA; investigue se for mais de 2× mais lento que o native WebGPU na mesma GPU.

Logs: `bench_logs/qwen3-q4km_{cuda,vulkan,wgpu}_final.log`.

## Requisitos do sistema

- Toolchain Rust com workspace `edition = "2021"`
- Para **Vulkan**: loader / ICD Vulkan funcional (Linux, Windows, Android)
- Para **WGPU**: adapter suportado pelo `wgpu` (Vulkan, DX12, Metal ou WebGPU no browser)
- CUDA opcional para paridade diferencial e baselines de SLO
- Python 3.x para `scripts/backend_parity_audit.py` e `bench_examples.py`

## Licença

O código neste repositório segue o dual licensing do Candle upstream: [Apache-2.0](./LICENSE-APACHE) e [MIT](./LICENSE-MIT).

Os pesos dos modelos upstream mantêm as licenças e restrições originais.
