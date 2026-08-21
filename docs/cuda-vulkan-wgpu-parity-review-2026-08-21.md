# Ревью паритета CUDA ↔ Vulkan ↔ WGPU: финал волны 2026-08-21 (два исправленных correctness-бага)

Дата: 2026-08-21 · Ветка: `wgpu/vulkan` · HEAD: `3935c343` · Среда: RTX 3060, Windows, ash 0.38 (Vulkan), wgpu 29.0.4.
Предыдущие документы: `docs/cuda-vulkan-wgpu-parity-review-2026-08-19.md` (baseline `54451acc`), `docs/perf-memory-audit-2026-08-19.md` (фиксы `d231ec93`, `ae86a0e2` — подтверждены на HEAD).

## 1. Резюме (TL;DR)

**Полного паритета с CUDA по-прежнему нет, но после этой волны все ~33 метода `BackendStorage` выполнимы на обоих GPU-бекендах на дефолтных dtype** (F32/F16/BF16 + U8/U32/I64/I32 там, где операция имеет смысл). Оставшийся разрыв — это (а) несколько f32-хабов/эмуляций вместо нативных CUDA-путей и (б) два мелких dtype-гэпа, один из которых — паритет с самой CUDA.

Главная ценность этой волны: **найдены и исправлены два реальных correctness-бага** (молчаливая порча данных, не ловившаяся стандартными тестами), а также классифицированы все оставшиеся падения дифференциального сюита `backend_parity_diff` — среди них больше нет неизвестных.

| Прогон (после всех фиксов) | Результат |
|---|---|
| candle-core lib, vulkan | 48 passed / 0 failed / 15 ignored |
| candle-core lib, wgpu | 108 passed / 0 failed |
| `cargo clippy --all-targets -- -D warnings` | vulkan: чисто · wgpu: чисто |
| backend_parity_diff, vulkan | 5 passed / 4 failed — все падения из известного списка, **новых нет** (было 5 функций) |
| backend_parity_diff, wgpu | 5 passed / 4 failed — все падения из известного списка, **новых нет** (было 5 функций) |
| Изолированный argsort-репро (dtype/знаки/дубли/asc+desc/large) | 18/18 групп зелёные |

## 2. Что изменилось с ревью 2026-08-19

- CUDA-стек не менялся с `54451acc` → матрица op×dtype из §3 прошлого ревью валидна и пере-проверена на HEAD.
- Закрытия волн 2026-08-20 (§10–11 прошлого документа: F64 native VK, F8E4M3 VK, scatter_add ext, wgpu int/reduce, native F16/BF16 conv/pool/upsample wgpu, flash_attn_varlen/paged) — подтверждены на HEAD.
- Эта волна: два исправленных correctness-бага (§3), классификация падений parity_diff (§6–7), подтверждение отсутствия утечек (§8).

## 3. Два исправленных correctness-бага

### 3.1. wgpu `powf`/`elu` F16 — молчаливая порча данных (ИСПРАВЛЕНО)

- **Симптом**: `backend_parity_diff::diff_unary`: `gpu=-1608` при `cpu=3.2`. Изолированный репро: 8 из 9 значений — мусор. Стандартные lib-тесты это не ловили (F16-покрытие powf там отсутствовало).
- **Корень**: `custom_unary_wgsl` (wgpu_backend.rs ~4794) жёстко эмитит `array<f32>`-storage. При включённом `SHADER_F16` F16-тензоры уходили в generic-путь `run_unary_like` без конвертации — f16-буфер интерпретировался как f32-слова (тихая порча, без валидационной ошибки).
- **Фикс** (wgpu_backend.rs ~12505 `powf`, ~12507 `elu`): `powf` F16/F64 → f32-хаб (`materialize_to_f32` → op → `to_dtype`); `elu(alpha≠1)` F16 без `SHADER_F16` → f32-хаб. BF16 уже шёл через `bf16_unary_via_f32`.
- **Верификация**: wgpu lib 108/108; clippy `-D warnings` чисто; parity_diff: powf/elu исчезли из падений; попутно перестал падать `cmp F64` (5→4 функций).

### 3.2. Vulkan `arg_sort_last_dim` F32 — неверный порядок на смешанных знаках + нестабильные ties у i64/f64 (ИСПРАВЛЕНО)

- **Симптом**: `backend_parity_diff`: строки результата выглядели «ротированными» (3×5: gpu row *i* = cpu row *i+1*); 1D-случай корректен. На первый взгляд указывало на multi-row dispatch `(1, nrows, 1)` — **ложный след**.
- **Реальный корень**: детальный дамп показал, что каждая выходная строка — валидный argsort *смещённой* исходной строки, а паттерн был периодичен (период 3 на LCG-фикстуре) → data-dependent comparator-баг. В `shaders/argsort.comp` (введён коммитом `e9104aa7` «CPU-stable argsort ties») orderable-ключ F32 (`bits ^ 0x80000000` для позитивов, `~bits` для негативов) корректен **только как uint**, но переменные объявлены `int` и сравнивались знаково → все положительные float получают выставленный старший бит и «становятся меньше» всех отрицательных. Итоговый порядок = `[позитивы↑, негативы↑]` — ротация корректного порядка на число негативов в строке, что на периодических фикстурах маскировалось под ротацию строк.
- **Того же семейства** (найдены репро до фикса): `argsort_i64`/`argsort_f64` и все large-варианты — без stable tie-break (6/6 строк расходились с CPU на дубликатах); DESC реализован reverse-write (ломает стабильность ties).
- **Фикс**: 7 шейдеров переведены на единый компаратор — `shaders/argsort.comp`, `shaders/argsort_large.comp`, `candle-shaders/argsort_i64.comp`, `argsort_f64.comp`, `argsort_large_u32.comp`, `argsort_large_i64.comp`, `argsort_large_f64.comp`:
  - F32-ключи сравниваются как `uint`;
  - stable tie-break по возрастанию исходного индекса в **обоих** направлениях (соответствие CPU `sort_by` stability);
  - DESC — инверсией ключа в компараторе (reverse-write убран, write-back прямой).
- **Верификация**: изолированный репро 18/18 групп — f32/f16(через хаб)/u32/i64/f64, дубликаты, asc+desc, small (ncols 4–16) и large-пути (ncols 1025/1500/2048, Vulkan memory model на RTX 3060); vulkan lib 48/48; clippy чисто; parity_diff: argsort-падения исчезли.

## 4. Сводная матрица op×dtype (статус после этой волны)

Полная матрица — §3 документа 2026-08-19 (CUDA-стек не менялся, валидна). Здесь — только дельта и оставшиеся отличия. Обозначения: **native** = собственный шейдер/ядро, **hub** = через F32-конвертацию (`gpu_resident_via_f32`/`cuda_parity_conv_via_f32`), **err** = UnsupportedDTypeForOp.

| Операция | CUDA | Vulkan | WGPU |
|---|---|---|---|
| matmul F16/BF16 | native (cublas) | hub (F32-ядро быстрее wgpu: 0.31 vs 0.63 мс) | hub |
| cmp (eq/ne/lt/…) F64 | native | native | native |
| cmp F8E4M3 / BF16 (VK) | native | hub | native/emulated |
| cmp I16/I32 (VK) | native | hub | native |
| reduce I16 | native | hub (через F32) | **err** (гэп) |
| unary F8E4M3 | native (arch≥890) | hub | **отсутствует** (гэп) |
| rand F16/BF16 | native | hub | native |
| argsort все dtype | native | **native + stable ties** (эта волна) | native |
| scatter_add I16/I32 | err (нет инстанциаций) | err — **паритет с CUDA** | err |
| scatter_add F8E4M3 | native (arch≥890) | err/хаб (мелкий гэп) | hub |
| to_dtype I16/I32 (VK) | err (mod.rs:1669-1671) | err — **паритет с CUDA** | native (emulated) |
| powf/elu F16/F64 | native | native | **native/hub — фикс этой волны** |

## 5. Оставшиеся гэпы до полного паритета (по убыванию приоритета)

1. **wgpu reduce I16** — ошибка (int-reduce покрывает U8/U32/I64/I32; у CUDA есть). Малый объём работы по аналогии с существующими int-reduce.
2. **wgpu unary F8E4M3** — отсутствует полностью (у VK есть через хаб). Дешёвый вариант: направить через f32-хаб как на VK.
3. **Vulkan scatter_add F8E4M3** — у CUDA native на arch≥890; тривиально закрывается хабом.
4. **Нативизация хабов** (перф, не корректность): matmul F16/BF16 wgpu, cmp BF16/F8E4M3 VK, rand F16/BF16 VK — каждый хаб = 2 конверсии + доп. аллокации.
5. **GGUF-квантование**: `GgmlDType` — 15 значений, нет IQ-серии/mxfp4/nvfp4 (CUDA поддерживает на новых arch); `quantize` у `QWgpuStorage`/`QVulkanStorage` — CPU round-trip; `supports_bf16` = false у Wgpu/Vulkan (`device.rs:367-372`).
6. **Перф-мелочь**: wgpu `flash_attn_varlen` — per-call `create_buffer` для params (~wgpu_backend.rs:8171); wgpu `sum_rows` — per-call `create_buffer` (~12619) →候选 пул параметров-буферов.

## 6. Харнесс-дефекты `backend_parity_diff` (НЕ баги бекендов)

Класс один: `check_vs_cpu` сравнивает по **исходному** dtype тензора, а не по dtype результата операции:

| Тест | Бекенд | Строк | Суть |
|---|---|---|---|
| `cmp` F64 | vulkan | 18 | `to_vec1::<f64>` на U8-результате cmp |
| `to_dtype→I64` | wgpu | 3 | `to_vec1::<f32>` на I64-результате |
| `argmax/argmin` F64 | wgpu | 4 | `to_vec1::<f64>` на U32-результате |

Изолированные репро конверсий подтвердили корректность самих операций. **Рекомендация**: сравнивать по dtype результата (явный mapping op→result_dtype в харнессе). До фикса харнесса эти падения останутся и будут зашумлять сигнал.

## 7. Толерансные различия (семантика, не баги)

| Кейс | Величина | Причина |
|---|---|---|
| `sin`/`cos`/`gelu` F32/special | до ~0.9 | NaN/Inf-семантика (gpu=-inf vs cpu=NaN) |
| `gelu_erf` F32/F64 | ~2.2e-4 систематически (20 падений wgpu, 2 vulkan) | erf-аппроксимация в шейдерах vs libm на CPU |
| `matmul` F16/square8 | 1.56e-2 > 1e-2 | F16-округление входов при F32-аккумуляции |
| `sum_dim` F16/BF16 llm | 6.25e-2 / 2.5e-1 | длинные суммы в f32-хабе с F16/BF16-округлением входов |

**Рекомендация**: expected-mismatch конфигурация или dtype-aware толерансы в сьюте, чтобы зелёный прогон означал именно отсутствие регрессий.

## 8. Перф и память (подтверждено на HEAD)

- **Утечек памяти нет**: deferred buffer frees (`MAX_DEFERRED_BUFFER_FREES=256`, vulkan_backend.rs:1371), staging pool `acquire_staging_buffer` (1889), `rustc_hash` FxHashMap pipeline cache (line 11) — на месте и работают.
- VK matmul F32 512×1024×1024: **0.31 мс** vs wgpu **0.63 мс** (VK ~2× быстрее).
- F32-хаб систематически дороже нативного пути — главные кандидаты на нативизацию перечислены в §5.4.

## 9. Следующие шаги

1. Фикс харнесса parity_diff (§6) — снимет 25 ложных FAIL-строк и сделает сигнал сьюта чистым.
2. Закрыть два мелких функциональных гэпа: wgpu reduce I16, wgpu unary F8E4M3 (через хаб).
3. Expected-mismatch/dtype-толерансы для кейсов §7.
4. Перф-волна: пул param-буферов wgpu (flash_attn_varlen, sum_rows); нативные пути для matmul F16/BF16 wgpu.
5. Расширение GGUF (IQ/mxfp4/nvfp4) — отдельная большая задача, приоритет ниже.

---
*Все утверждения о коде проверены на HEAD 3935c343; тестовые прогоны — RTX 3060, release, 2026-08-21. Фиксы этой волны: wgpu_backend.rs (powf/elu F16), 7 argsort-шейдеров candle-vulkan-kernels.*
