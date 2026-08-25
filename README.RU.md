<p align="left">
  <a href="README.md"><img src="https://img.shields.io/badge/English-232323" alt="English"></a>
  <a href="README.RU.md"><img src="https://img.shields.io/badge/Русский-D65C5C" alt="Русский"></a>
  <a href="README.PT_BR.md"><img src="https://img.shields.io/badge/Português_BR-232323" alt="Português"></a>
</p>

---

<p align="center">
  <b>Форк Candle с native Vulkan и WGPU / WebGPU бэкендами.</b><br>
  Паритет с CUDA для инференса на Linux, Windows, macOS, Android и WASM.
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

## Матрица платформ

| Backend | Feature | Linux | Windows | macOS | Android | WASM |
|---------|---------|:-----:|:-------:|:-----:|:-------:|:----:|
| CPU | (default) | ✅ | ✅ | ✅ | ✅ | ✅ |
| CUDA | `cuda` | ✅ | ✅ | ❌ | ❌ | ❌ |
| Metal | `metal` | ❌ | ❌ | ✅ | ❌ | ❌ |
| WGPU | `wgpu` | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vulkan | `vulkan` | ✅ | ✅ | ❌ | ✅ | ❌ |

Бэкенды **CPU** и **CUDA** синхронизируются напрямую с [huggingface/candle](https://github.com/huggingface/candle) `main`. Из них ничего не удалено; этот форк только добавляет Vulkan и WGPU поверх upstream.

## Соответствие upstream

| Форк ([FerrisMind/candle](https://github.com/FerrisMind/candle)) | Upstream ([huggingface/candle](https://github.com/huggingface/candle)) |
|------------------------------------------------------------------|------------------------------------------------------------------------|
| Ветка `wgpu/vulkan` | `main` |
| Крейты форка **0.0.174** (`candle-core` / `candle-nn` / `candle-transformers` / `candle-examples` / `candle-*-kernels` для Vulkan и WGPU); без изменений — **0.11.0** | Candle **0.11.0** |
| Последний sync CPU/CUDA [`b3e5b40f`](https://github.com/FerrisMind/candle/commit/b3e5b40f) (2026-08-17) | Tip [`162b59b9`](https://github.com/FerrisMind/candle/commit/162b59b9) (#3892) |
| Только в форке | Native **Vulkan** + **WGPU / WebGPU** |

CPU и CUDA синхронизируются с upstream `main` (ничего не удалено). Vulkan и WGPU есть только в этом форке.

## Содержание

- [Матрица платформ](#матрица-платформ)
- [Соответствие upstream](#соответствие-upstream)
- [Что это?](#что-это)
- [Ключевые возможности](#ключевые-возможности)
- [Структура репозитория](#структура-репозитория)
- [Быстрый старт](#быстрый-старт)
- [Производительность бэкендов](#производительность-бэкендов)
- [Системные требования](#системные-требования)
- [Лицензия](#лицензия)

## Что это?

Это форк [huggingface/candle](https://github.com/huggingface/candle) на ветке по умолчанию [`wgpu/vulkan`](https://github.com/FerrisMind/candle/tree/wgpu/vulkan). Upstream остаётся источником tensor API, моделей и путей CPU / CUDA / Metal — эти бэкенды синхронизируются с `main` и не вырезаются. В этой ветке добавлены и доработаны:

- **native Vulkan** compute-бэкенд (`ash` + SPIR-V)
- **WGPU / WebGPU** compute-бэкенд (`wgpu` + WGSL)
- parity-документация, smoke-тесты и CUDA differential coverage для этих бэкендов

Паритет ведётся по трём отдельным профилям (результаты не смешивать):

| Профиль | Смысл |
|---------|-------|
| Native Vulkan | Прямой Vulkan / SPIR-V |
| Native WebGPU | Native `wgpu` с runtime feature detection |
| Portable WebGPU | Browser / WASM-safe WGSL; без native-only заявлений |

Нормативные документы:

- [`docs/backend-parity-spec.md`](./docs/backend-parity-spec.md)
- [`docs/backend-parity.md`](./docs/backend-parity.md)
- [`docs/backend-parity-manifest.json`](./docs/backend-parity-manifest.json)

## Ключевые возможности

- Cargo features `vulkan` и `wgpu` включают новые GPU-бэкенды.
- Выбор устройства: `CANDLE_DEVICE=vulkan|wgpu` (также `cuda`, `metal`, `cpu`).
- Выбор адаптера: `CANDLE_VULKAN_DEVICE_NAME`, `CANDLE_WGPU_ADAPTER_NAME`.
- Override API для WGPU: `WGPU_BACKEND=vulkan|dx12|metal|gl`.
- Без скрытого CPU compute под видом GPU и без silent cast dtype → F32.
- Static parity audit, smoke-тесты, CUDA differential matrix, fallback audit и bench harness для examples.

## Структура репозитория

| Путь | Назначение |
|------|------------|
| `candle-core` | Tensor API, devices, Vulkan / WGPU storage и ops |
| `candle-vulkan-kernels` | SPIR-V compute shaders |
| `candle-wgpu-kernels` | WGSL compute shaders |
| `candle-nn` | Слои (включая MoE и flash-attn dispatch) |
| `candle-transformers` | Реализации моделей |
| `candle-examples` | Примеры для e2e покрытия бэкендов |
| `docs/` | Parity-спецификация, манифест и evidence |
| `scripts/` | Parity audit и связанный tooling |
| `bench_examples.py` | Multi-backend harness по throughput examples |

## Быстрый старт

### Запуск example на Vulkan или WGPU

```powershell
$env:CANDLE_DEVICE = "vulkan"   # или "wgpu"
cargo run -p candle-examples --release --features vulkan --example quantized-qwen3 -- --model <path-to-gguf>
cargo run -p candle-examples --release --features wgpu --example quantized-qwen3 -- --model <path-to-gguf>
```

### Parity-проверки

```powershell
python scripts/backend_parity_audit.py

cargo test -p candle-core --features vulkan --test backend_smoke_tests
cargo test -p candle-core --features wgpu --test backend_smoke_tests

# CUDA differential matrix (нужны GPU)
$env:CANDLE_REQUIRE_CUDA_TEST_DEVICE = "1"
$env:CANDLE_REQUIRE_VULKAN_TEST_DEVICE = "1"
$env:CANDLE_REQUIRE_WGPU_TEST_DEVICE = "1"
cargo test -p candle-core --features "cuda,vulkan,wgpu" --test gpu_parity_matrix_tests

cargo run -p candle-core --release --features "vulkan,wgpu" --example fallback_runtime_audit
cargo run -p candle-core --release --features "cuda,vulkan,wgpu" --example backend_parity_microbench -- --suite

python bench_examples.py --models-root <models-root> --backend cuda --backend vulkan --backend wgpu
```

## Производительность бэкендов

End-to-end throughput для [`quantized-qwen3`](./candle-examples/examples/quantized-qwen3/) (**Qwen3-0.6B-GGUF Q4_K_M**, release, CUDA baseline в той же сессии).

Железо: **RTX 3060 12 GB**, **Ryzen 7 3700X**, **64 GB DDR4**. `%CUDA` — доля от CUDA; Min / Normal / Goal — SLO tiers из [`bench_examples.py`](./bench_examples.py).

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

CUDA baseline (та же сессия): tg128 60.09, tg256 59.57, pp512 58.79, pp1024 56.74, pp2048 58.78, pp4096 43.91 tok/s.

Release SLO (end-to-end vs CUDA): **Vulkan ≤ 15% медленнее** (stretch 10%); **native WebGPU ≤ 30% медленнее** (stretch 20%). Для portable WebGPU фиксированного % от CUDA нет; расследовать, если >2× медленнее native WebGPU на том же GPU.

Логи: `bench_logs/qwen3-q4km_{cuda,vulkan,wgpu}_final.log`.

## Системные требования

- Rust toolchain с workspace `edition = "2021"`
- Для **Vulkan**: рабочий Vulkan loader / ICD (Linux, Windows, Android)
- Для **WGPU**: адаптер, поддерживаемый `wgpu` (Vulkan, DX12, Metal или browser WebGPU)
- Опционально CUDA — для differential parity и SLO baseline
- Python 3.x для `scripts/backend_parity_audit.py` и `bench_examples.py`

## Лицензия

Код в этом репозитории — dual license как у upstream Candle: [Apache-2.0](./LICENSE-APACHE) и [MIT](./LICENSE-MIT).

Веса upstream-моделей сохраняют исходные лицензии и ограничения.
