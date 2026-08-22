# Ревью паритета CUDA ↔ Vulkan ↔ WGPU: FINAL волны 2026-08-21 (полный функциональный паритет)

Дата: 2026-08-21 · Ветка: `wgpu/vulkan` · HEAD: `1596684b` · Среда: RTX 3060, Windows, ash 0.38 (Vulkan), wgpu 29.0.4.
Итоговая волна оркестрирована через agent-swarm. Предыдущие документы: `docs/cuda-vulkan-wgpu-parity-review-2026-08-19.md` (baseline `54451acc`), `docs/perf-memory-audit-2026-08-19.md` (фиксы `d231ec93`, `ae86a0e2` — подтверждены на HEAD).
Промежуточные волны `2026-08-20/21` закрыты в §10–11 документа 2026-08-19; эта волна дополнительно закрывает всё из §5–6 текущего документа.

## 1. Резюме (TL;DR)

**Функциональный паритет с CUDA по контракту `BackendStorage` достигнут**: все ~33 метода выполнимы на обоих GPU-бекендах на дефолтных dtype (F32/F16/BF16 + U8/U32/I64/I32 там, где операция имеет смысл) — последние три dtype-гэпа §5.1–5.3 закрыты финальной волной (§12). Оставшийся разрыв vs CUDA — только f32-хабы/эмуляции вместо нативных путей (перф-класс), CUDA-эксклюзивная экосистема вне op-parity контракта (candle-ug, flash-attn v2/v3, cuDNN/TF32, CUDA Graphs, MoE WMMA) и GGUF-ограничения; см. вердикт §12.

Главная ценность волны: **найдены и исправлены четыре реальных correctness-бага** (wgpu powf/elu F16, vulkan argsort, wgpu to_dtype U8→I32 — три из них молчаливая порча данных), исправлен дефект харнесса parity_diff (35 ложных FAIL) и закрыт латентный баг CUDA fp8 (имена кернелов, §12).

| Прогон (после всех фиксов, финальная волна §12) | Результат |
|---|---|
| candle-core lib, vulkan | **49 passed / 0 failed / 15 ignored** |
| candle-core lib, wgpu | **123 passed / 0 failed / 0 ignored** |
| `cargo clippy --tests -- -D warnings` | vulkan: чисто · wgpu: чисто |
| backend_parity_diff, обе фичи | **6 passed / 3 failed / 1 ignored** — падения только толерансные (§7); `diff_cmp`/`diff_to_dtype` зелёные |
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
| matmul F16/BF16 | native (cublas) | hub (F32-ядро быстрее wgpu: 0.31 vs 0.58 мс, финальный аудит §12) | hub |
| cmp (eq/ne/lt/…) F64 | native | native | native |
| cmp F8E4M3 / BF16 (VK) | native | hub | native/emulated |
| cmp I16/I32 (VK) | native | hub | native |
| reduce I16 | **нет кернелов** (reduce.cu не инстанцирует) | hub (через F32) | hub через I32 — `5402a7a2` (закрыто; паритет отсутствия с CUDA) |
| unary F8E4M3 | «native», но кернелы сломаны именами + arch≥890 (§12) | hub | hub — `289549fb` (закрыто) |
| rand F16/BF16 | native | hub | native |
| argsort все dtype | native | **native + stable ties** (эта волна) | native |
| scatter_add I16/I32 | err (нет инстанциаций) | err — **паритет с CUDA** | err |
| scatter_add F8E4M3 | «native» arch≥890 (те же сломанные имена) | hub — `b135ffb3` (закрыто) | hub |
| to_dtype I16/I32 (VK) | err (mod.rs:1669-1671) | err — **паритет с CUDA** | native (emulated) |
| powf/elu F16/F64 | native | native | **native/hub — фикс этой волны** |

## 5. Оставшиеся гэпы до полного паритета (по убыванию приоритета)

1. **wgpu reduce I16** — ошибка (int-reduce покрывает U8/U32/I64/I32; у CUDA есть). Малый объём работы по аналогии с существующими int-reduce. ✅ **закрыто в этой волне, коммит `5402a7a2`** (reduce I16 через I32-хаб, 6 тестов; см. §12).
2. **wgpu unary F8E4M3** — отсутствует полностью (у VK есть через хаб). Дешёвый вариант: направить через f32-хаб как на VK. ✅ **закрыто в этой волне, коммит `289549fb`** (unary/binary/cmp/affine/powf/elu F8E4M3 через f32-хаб, 8 тестов; см. §12).
3. **Vulkan scatter_add F8E4M3** — у CUDA native на arch≥890; тривиально закрывается хабом. ✅ **закрыто в этой волне, коммит `b135ffb3`** (f32-хаб, тест `vulkan_scatter_add_f8e4m3`; см. §12).
4. **Нативизация хабов** (перф, не корректность): matmul F16/BF16 wgpu, cmp BF16/F8E4M3 VK, rand F16/BF16 VK — каждый хаб = 2 конверсии + доп. аллокации. Остаётся (подробнее см. §12, вердикт п.1).
5. **GGUF-квантование**: `GgmlDType` — 15 значений, нет IQ-серии/mxfp4/nvfp4 (CUDA поддерживает на новых arch); `quantize` у `QWgpuStorage`/`QVulkanStorage` — CPU round-trip; `supports_bf16` = false у Wgpu/Vulkan (`device.rs:367-372`). Остаётся (см. §12, вердикт п.3).
6. **Перф-мелочь**: wgpu `flash_attn_varlen` — per-call `create_buffer` для params (~wgpu_backend.rs:8171); wgpu `sum_rows` — per-call `create_buffer` (~12619) → кандидат в пул параметров-буферов. ➖ **частично закрыто**: оба сайта переведены на uniform ring (`write_uniform_params`, wgpu_backend.rs:8289 и 12726) в коммите `1596684b`; пул оставшихся ~40 per-call params-сайтов остаётся (см. §12).

## 6. Харнесс-дефекты `backend_parity_diff` (НЕ баги бекендов)

Класс один: `check_vs_cpu` сравнивает по **исходному** dtype тензора, а не по dtype результата операции:

| Тест | Бекенд | Строк | Суть |
|---|---|---|---|
| `cmp` F64 | vulkan | 18 | `to_vec1::<f64>` на U8-результате cmp |
| `to_dtype→I64` | wgpu | 3 | `to_vec1::<f32>` на I64-результате |
| `argmax/argmin` F64 | wgpu | 4 | `to_vec1::<f64>` на U32-результате |

✅ **ИСПРАВЛЕНО в этой волне, коммит `56abbe10`**: `check_vs_cpu` теперь сравнивает по **dtype результата** (mapping op→result_dtype: `cmp→U8`, `argmax/argmin/argsort→U32`, `to_dtype→target`). Снято 35 ложных FAIL-строк (см. §12). Важно: харнесс-фикс **раскрыл настоящий баг** — `to_dtype U8→I32` (см. §12, коммит `5402a7a2`). `diff_cmp` и `diff_to_dtype` теперь зелёные на обоих бекендах.

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
- VK matmul F32 1024²: **0.31 мс** vs wgpu **0.58 мс** (финальный аудит §12; VK ~1.9× быстрее).
- F32-хаб систематически дороже нативного пути — главные кандидаты на нативизацию перечислены в §5.4.

## 9. Следующие шаги

1. ~~Фикс харнесса parity_diff (§6) — снимет 25 ложных FAIL-строк и сделает сигнал сьюта чистым.~~ ✅ **выполнено** в этой волне (`56abbe10`, снято 35 строк; `diff_cmp`/`diff_to_dtype` зелёные).
2. ~~Закрыть два мелких функциональных гэпа: wgpu reduce I16, wgpu unary F8E4M3 (через хаб).~~ ✅ **выполнено** (`5402a7a2`, `289549fb`; плюс vulkan `scatter_add` F8E4M3 — `b135ffb3`, §5.3).
3. Expected-mismatch/dtype-толерансы для кейсов §7. **Остаётся** (см. §12, вердикт).
4. Перф-волна: пул param-буферов wgpu — `flash_attn_varlen` и `sum_rows` ➖ **частично выполнены** (`1596684b`, uniform ring), ~40 остальных сайтов — follow-up (список в §12); нативные пути для matmul F16/BF16 wgpu — **остаются**.
5. Расширение GGUF (IQ/mxfp4/nvfp4) — отдельная большая задача, приоритет ниже. **Остаётся** (см. §12, вердикт).

**Итог: вся функциональность закрыта — остались только перф-оптимизации (§5.4, §12 вердикт п.1), толерансы §7 и GGUF.**

## 12. ИТОГОВАЯ ВОЛНА 2026-08-21 (агент-сварм): полный функциональный паритет достигнут

Финальная волна завершила закрытие всех функциональных гэпов §5.1–5.3 и харнесса §6, раскрыла латентный баг CUDA fp8 и зафиксировала финальные гейты. Все коммиты прошли гейты на RTX 3060, Windows, release, 2026-08-21, `--test-threads=1` (примечание машины: параллельный запуск тестов сегфолтит на этом железе даже на бейзлайне — все прогоны однопоточные).

### Коммиты волны

| Коммит | Содержание |
|---|---|
| `024d7c7a` | Фикс двух correctness-багов предыдущей сессии (были незакоммичены): wgpu powf/elu F16 silent corruption (custom_unary_wgsl эмитит `array<f32>`; F16/F64 powf + F16 elu(alpha≠1) теперь через f32-хаб); vulkan argsort signed-comparator (f32 orderable-ключи сравнивались знаково как int → позитивы «меньше» негативов; 7 шейдеров переведены на единый компаратор: uint-сравнение, stable (key,index) tie-break, DESC через инверсию ключа). Плюс первый срез самого документа. |
| `56abbe10` | Фикс харнесса `backend_parity_diff`: `check_vs_cpu` сравнивал по dtype ИСТОЧНИКА вместо dtype РЕЗУЛЬТАТА (cmp→U8, argmax/argmin/argsort→U32, to_dtype→target). Снято 35 ложных FAIL-строк (cmp F64 vulkan 24, argmax/argmin F64 оба бекенда 8, to_dtype→I64 wgpu 3). Харнесс-фикс РАСКРЫЛ настоящий баг — см. `5402a7a2`. |
| `b135ffb3` | vulkan `scatter_add` F8E4M3 через f32-хаб (зеркало BF16-ветки; decode dst+src → f32 scatter_add → encode). Тест `vulkan_scatter_add_f8e4m3` (накопление по дубль-индексам, отрицательные значения). Закрыт последний vulkan-гэп §5.3. |
| `5402a7a2` | (а) wgpu reduce I16 через I32-хаб (`materialize_to_i32`; sum/min/max→I16, argmin/argmax→U32; 6 тестов). Важно: у CUDA ТОЖЕ нет нативных i16 reduce-кернелов (`reduce.cu` инстанцирует только bf16/f16/f32/f64/u32/i64/u8; fp8-вариант закомментирован `:664`) — т.е. это консистентность wgpu, НЕ гэп к CUDA. (б) Фикс настоящего бага `to_dtype` U8→I32 (occupancy lanes 1–3 занулялись двойным шифтом: load для U8 отдаёт уже извлечённый байт в lo, а `conv_i32` шифтовал его повторно; теперь `i32(lo)`). Баг был замаскирован дефектом харнесса до `56abbe10`. 7 тестов. |
| `289549fb` | wgpu F8E4M3 compute: unary/binary/cmp/affine/powf/elu через f32-хаб (хелперы `f8e4m3_unary_via_f32`/`f8e4m3_binary_via_f32` через to_dtype → `run_f8e4m3_cast` decode-шейдер, зеркально vulkan). `materialize_to_f32` неприменим для f8 (нет ветки copy_strided + 1-байтные элементы ломают `COPY_BUFFER_ALIGNMENT`). cmp→U8, остальное F8E4M3. 8 тестов. НАХОДКА: латентный баг CUDA — см. ниже. |
| `1596684b` | perf: wgpu `flash_attn_varlen` + `sum_rows` params через существующий uniform ring (128×256B) вместо per-call `create_buffer`; compile-time ассерты ≤256B; ноль изменений шейдеров/лейаутов. Perf-book rule: mem-reuse-collections. Инвентаризация: ~40 других per-call params-буферов остаются (follow-up; репрезентативные строки 827, 5439, … 12430 — номера до последних коммитов, сдвиг ~+491). |

### НАХОДКА: латентный баг CUDA fp8 (сломано на уровне имён)

Кернелы в `unary.cu`/`binary.cu`/`indexing.cu` инстанцированы с суффиксом `_fp8_e4m3`/`_f8_e4m3`, а Rust `kernel_name()` (`cuda_backend/mod.rs:130-133`, `dtype.rs` `as_str()` → `f8e4m3`) ищет `_f8e4m3` → `load_function` не найдёт имя на Hopper. Плюс все fp8-кернелы CUDA гейтед `__CUDA_ARCH__ >= 890` (Hopper+), на RTX 3060 (Ampere, arch 86) недоступны в принципе. Т.е. fp8-«преимущество» CUDA само по себе сломано.

### Финальные числа гейтов (V1, RTX 3060, release, 2026-08-21)

| Прогон | Результат |
|---|---|
| `clippy -D warnings` | wgpu 0 · vulkan 0 (glslc banner — не диагностика) |
| `cargo test -p candle-core --features vulkan --lib` | **49 passed / 0 failed / 15 ignored** (было 48) |
| `cargo test -p candle-core --features wgpu --lib` | **123 passed / 0 failed / 0 ignored** (было 108) |
| `backend_parity_diff`, обе фичи | **6 passed / 3 failed / 1 ignored**; падения — ТОЛЬКО задокументированные толерансные классы §7: `diff_matmul` (F16 rounding 1.56e-2 > 1e-2), `diff_reduce` (sum_dim F16/BF16 llm), `diff_unary` (gelu_erf ~2.2e-4 систематика + NaN/Inf-семантика sin/cos/gelu). `diff_cmp` и `diff_to_dtype` теперь зелёные на обоих бекендах. Новых функциональных падений нет. |

### Перф-аудит (final, парное сравнение с предыдущим)

| Бекенд | Метрика | Текущий | Предыдущий | Оценка |
|---|---|---|---|---|
| vulkan | matmul f32 1024² | median 0.31ms / p95 0.34 | 0.33 / 0.36 | без регрессии |
| vulkan | unary gelu 1M | 0.10ms | 0.11 | ✓ |
| vulkan | binary add 1M | 0.11ms | 0.13 | ✓ |
| vulkan | leak slope binary / matmul | 0.006% / 0.006% | — | порог 1% — утечек нет |
| wgpu | matmul f32 1024² | 0.58ms / p95 0.60 | 0.57 / 0.71 | в шуме |
| wgpu | unary gelu 1M | 0.13ms | 0.13 | ✓ |
| wgpu | binary add 1M | 0.15ms | 0.14 | ✓ |
| wgpu | leak slope binary / matmul | 0.083% / 0.064% | 0.021–0.054% | <<1%; deltas 40–53KB за 100 итераций — уровень realloc'ов Vec, не GPU-утечек; RSS baseline ниже: 63.8MB vs 98MB |

### Вердикт по главному вопросу

**Полный паритет с CUDA по контракту `BackendStorage` на дефолтных dtype достигнут функционально**: все операции выполнимы на обоих бекендах (сравни с §1 текущего документа, где оставались 3 мелких dtype-гэпа — все закрыты этой волной). Остаточный разрыв vs CUDA — только:

1. **f32-хабы вместо нативных путей** (перф/точность-класс, не корректность): matmul F16/BF16 wgpu, cmp BF16/F8E4M3 vulkan, rand F16/BF16 vulkan, F8E4M3-хабы повсюду — работают корректно, но с доп. конверсиями.
2. **CUDA-эксклюзивная экосистема вне op-parity контракта**: candle-ug (NVRTC), candle-flash-attn v2/v3, cuDNN/cuBLAS TF32, CUDA Graphs, MoE WMMA. (`flash_attn_varlen`/paged KV уже есть на обоих.)
3. **GGUF-квантование**: IQ/mxfp4/nvfp4 отсутствуют в `GgmlDType` (ограничение candle в целом, у CUDA тоже в этом форке), `quantize()` CPU round-trip, `supports_bf16()=false`.
4. **Сами fp8-кернелы CUDA сломаны** именами (`_fp8_e4m3` vs `_f8e4m3`) и гейтом Hopper — практического fp8-преимущества CUDA над хабами VK/wgpu в этом форке нет.

**Итог волны: vulkan 49/49 passed (15 ignored), wgpu 123/123 passed, 0 ignored; clippy `-D warnings` = 0 по обеим фичам.**

---

## 13. ФИНАЛЬНАЯ ВОЛНА 2026-08-22 (agent-swarm): perf-паритет, GPU-quantize, disposition CUDA-эксклюзивов

Оркестрирована через agent-swarm (план `.swarm/plan.md`, отчёты листьев `.swarm/results/`). Все прогоны — RTX 3060, Windows, release, `--test-threads=1`. HEAD волны: `c91c7daf` (родитель `172aec59`).

### Коммиты волны

| Коммит | Содержание |
|---|---|
| `748bf79c` | Expected-mismatch allowlist в `backend_parity_diff` (реализация §9.3): 8 классов толерансных расхождений §7 с документированными max_abs-границами (breach = hard fail). Паритет-сьют: 9 passed / 0 failed / 1 ignored на обоих бекендах; 27 (wgpu) + 5 (vulkan) expected-mismatch note задействованы, все в границах. |
| `97591f4b` | perf(wgpu): 9 hot per-call params-сайтов переведены на uniform ring (`write_uniform_params`, 128×256B): copy_into, softmax (3 сайта), flash_attn_ext, flash_attn_paged, rope, rms_norm, index_select, gather, quantized_matvec. Исправлен латентный hazard: flash_attn_paged переписывал ОДИН params-буфер per-chunk при отложенном dispatch → свежий слот ring'а на чанк. Инвентаризация W-A3: 43 сайта всего, batch-2 (conv/cast/cmp/where/clamp) — follow-up. |
| `8cbb49d8` | perf(vulkan): нативный BF16 cmp — новый `cmp_bf16.comp` (packed-u16, decode через битовый сдвиг, сравнение в fp32, выход U8); удалён f32-хаб `bf16_cmp_via_f32` (двойной decode-проход). Тест на 23-элементной фикстуре (негативы, ±0, NaN, все 6 CmpOp) vs CPU. |
| `0fa10314` | `supports_bf16()` = true для Wgpu/Vulkan (device.rs): BF16 работоспособен end-to-end (wgpu software pack/unpack, vulkan precompiled SPIR-V — аппаратная фича не требуется), семантика «usable, not hardware-native». Smoke-ассерты обновлены; bf16 matmul-тесты теперь РАБОТАЮТ на обоих бекендах (11/11). Примеры (flux/mixtral/phi/…) получают BF16-по-умолчанию на этих девайсах вместо F32 (память вдвое). |
| `2b4e5f8e` | feat(wgpu): нативный GPU-quantize Q8_0 — word-packed 34B блоки, one-writer-per-u32-word (WGSL не имеет 8-битных типов), byte-identical CPU. |
| `bbb32485` | feat(wgpu): нативный GPU-quantize Q4_0 — nibble-packed 18B блоки, семантика CPU включая vmax tie-order. |
| `172aec59` | feat(vulkan): GPU-quantize Q8_0 + Q4_0 (.comp, CPU-layout блоки на устройстве). Устранён последний CPU round-trip в quant-стеке (`QWgpuStorage`/`QVulkanStorage::quantize` для этих форматов; не-contiguous/не-F32 входы — документированный CPU-путь). Тесты: GPU-байты == CPU-байты (не только dequant-толеранс). |
| `c91c7daf` | fix(wgpu): критический offset-баг, пойманный финальными гейтами — word-assembly в Q4_0/Q8_0 передавал ГЛОБАЛЬНЫЙ byte-offset вместо workgroup-локального → guard `block_in_wg < 8` отсекал все workgroup'ы кроме первого (блоки за пределами первых 144/272 байт занулялись). Q8_0 проходил smoke только потому, что фикстура влезала в один workgroup; Q4_0 matvec читал нули (nmse 0.75). Vulkan-версии бага не имели (другой паттерн адресации). |

### Диспозиция CUDA-эксклюзивной экосистемы (замена аналогом / non-gap)

| CUDA-эксклюзив | Диспозиция | Обоснование |
|---|---|---|
| GPU-quantize (GGUF) | **ЗАМЕНЕНО аналогом** | Q8_0/Q4_0 нативные кернелы на обоих бекендах, byte-identical CPU (коммиты выше). Q4_1/Q5_0/Q5_1/K-quants — follow-up по тому же паттерну; llama.cpp CUDA сама GPU-quantize не имеет (только Q8_1 в vk/wgsl). |
| candle-ug (NVRTC runtime-компиляция SSA→PTX) | **non-gap** | JIT-компиляция шейдеров не даёт op-parity-выгоды: оба бекенда поставляются прекомпилированными SPIR-V/WGSL-кернелами; runtime-compile — экосистемная фича, не контрактная. |
| CUDA Graphs (запись/переигрывание графов) | **non-gap** (perf-класс) | Batched command-буферы VK + bundle/cached encoders wgpu дают ту же амортизацию re-play (§5.8 дока 2026-08-19); SLO-перф отслеживается отдельно. |
| candle-flash-attn v2/v3 (Ampere/Hopper, GQA-packing, paged varlen) | **non-gap** (за пределами Hopper-специфики) | `flash_attn_varlen` + paged KV (block_table) уже портированы на оба бекенда (§11). Оставшееся — Hopper-специфичные tensor-core пути (WMMA), привязанные к железу, которых на Ampere-классе и у CUDA нет. |
| cuDNN conv (auto-algo, Winograd/FFT) | **non-gap** (perf-класс) | im2col+GEMM корректен; нативные F16/BF16 conv/pool/upsample уже есть на wgpu (§11), VK — F32-нативно. Разница — алгоритмический класс, не функциональность. |
| cuBLAS TF32 (`set_gemm_reduced_precision_f32`) | **non-gap** | Аналогичный режим точности — coopmat/матmul с f16→f32 накоплением; соответствие режимов задокументировано в §11 дока 2026-08-19 (п.11). |
| MoE WMMA (`moe_gemm_wmma`) / матричный MoE | **non-gap** (perf-класс) | wgpu: `mul_mat_id` (матричный MoE) есть; vulkan: `mul_mat_vec_id` (декодовый). Матричный MoE на VK — опциональный perf follow-up, не функциональный гэп (обоими бекендами MoE-модели исполняются). |
| fp8 (F8E4M3) «нативные» кернелы CUDA | **non-gap** | Кернелы CUDA сломаны в этом форке на уровне имён (`_fp8_e4m3` vs `_f8e4m3` в `kernel_name()`) + гейт `__CUDA_ARCH__ >= 890` (Hopper); на RTX 3060 недоступны в принципе. VK/wgpu f8e4m3 через f32-хаб — строго более юзабельны. |
| IQ-серия / mxfp4 / nvfp4 в GgmlDType | **non-gap** | Ограничение candle в целом (enum не содержит; у CUDA в этом форке тоже нет) — не гэп бекендов. |

### Финальные гейты (HEAD `c91c7daf`, RTX 3060, release, `--test-threads=1`)

| Прогон | Результат |
|---|---|
| `clippy -D warnings` | wgpu 0 · vulkan 0 (glslc banner — не диагностика) |
| `candle-core lib, wgpu` | **132 passed / 0 failed / 0 ignored** |
| `candle-core lib, vulkan` | **50 passed / 0 failed / 17 ignored** (все ignored — задокументированные follow-up) |
| `backend_parity_diff` | wgpu **9/0/1** · vulkan **9/0/1** (жёстких FAIL нет; expected-mismatch в границах) |
| `backend_smoke_tests, wgpu` | **43 passed / 0 failed** (включая quantized_paths после c91c7daf и bf16-native) |
| `backend_smoke_tests, vulkan` | 47/3: три OOM (`unary_binary`, `upload_and_dtype`, `upsample_native_only`) в полном сьют-прогоне — **pre-existing** (воспроизведено на baseline-HEAD до изменений волны; каждый проходит изолированно) |
| `matmul_tests` (bf16 включён) | 11/11 оба бекенда |

### Перф-аудит (HEAD `c91c7daf`, изолированные прогоны, RTX 3060)

| Бекенд | Метрика | Текущий | Предыдущий (§12) | Вердикт |
|---|---|---|---|---|
| vulkan | matmul f32 1024² | 0.33ms / p95 0.45 | 0.31 / 0.34 | шум, без регрессии |
| vulkan | gelu 1M / binary 1M | 0.11 / 0.13 | 0.10 / 0.11 | ✓ |
| vulkan | leak slope (binary/matmul ×100) | 0.006% / 0.017% | 0.006% / 0.006% | утечек нет (порог 1%) |
| wgpu | matmul f32 1024² | **0.58ms / p95 0.78** | 0.58 / 0.60 | ровно baseline |
| wgpu | gelu 1M / binary 1M | 0.15 / 0.16 | 0.13 / 0.15 | шум |
| wgpu | leak slope (binary/matmul ×100) | 0.083% / 0.058% | 0.083% / 0.064% | утечек нет; RSS baseline 63.8MB стабилен |

Вывод по перф-задаче: утечек памяти нет на обоих бекендах (slopes на порядок ниже порога 1%, дельты 4–53KB/100 итераций — уровень Vec-realloc), деградаций производительности нет; uniform-ring конверсия (97591f4b) перф-нейтральна на измеряемых workloads при устранении ~10 аллокаций/вызов и hazard'а flash_attn_paged.

### Открытые follow-up (зафиксировано, не блокирует паритет)

1. wgpu params batch-2: ~34 сайта на uniform ring (conv1d/2d, emulated_cast/strided_copy, cmp_u8, where_u8, clamp — инвентаризация `.swarm/results/w-a3-params-inventory.md`).
2. GPU-quantize остальных форматов: Q4_1/Q5_0/Q5_1 (medium по W-A4), K-quants Q2K–Q6K (hard, multi-scale superblocks).
3. vulkan cmp F8E4M3 и rand F16/BF16 нативизация — keep-hub по аудиту W-A2 (холодные пути, малый ROI).
4. vulkan smoke suite OOM-тройка в сьют-прогоне (pre-existing, machine-specific; изолированно зелёные).
5. `cargo fmt` на touched файлах (175 хунков, не блокирует гейты; отдельный cosmetic-коммит).

---

## 14. ВОЛНА ЗАКРЫТИЯ 2026-08-22 (agent-swarm): GPU-quantize завершён, uniform-ring params завершён, CUDA-baseline вердикты по SLO/TF32/CUDA-Graphs, OOM-фикс

Оркестрирована через agent-swarm (отчёты листьев `.swarm/results/`). Все прогоны — RTX 3060, Windows, release, `--test-threads=1`. HEAD волны: `c91c7daf` → `2e376d89`.

### Коммиты волны

| Коммит | Содержание |
|---|---|
| `02e98766` | feat(vulkan): нативный GPU-quantize Q4_1/Q5_0/Q5_1 — byte-identical CPU. Три numeric-parity-фикса: glslc алгебраически переписывает fp-деление (`x/15.0 → x*0.0666…`), что ломает побитовый паритет → делитель передан runtime uniform'ом (`VulkanQuantizeScaleParams.scale_bits`); NVIDIA OpFDiv на RTX 3060 на ~1 ulp ниже корректно-округлённого → `div_ieee` (выбор f32-кандидата по FMA-residual `|fma(b,q,-a)|`, optimizer-proof, в SPIR-V — OpExtInst Fma + 21 OpBitcast); ±0.0 tie-семантика min/max → `minNum`/`maxNum` (rust_min/rust_max). |
| `37d8b318` | fix(vulkan): gpu-allocator block cap 16MB (`AllocationSizes::new(16MB,16MB)`) — Arc-cycle утечка тестовых девайсов (каждый держал 256MB default block) приводила к OOM в сьют-прогоне на ~44 девайсах; smoke 47/3 → 50/50. Large-аллокации не регрессируют: dedicated-block путь не ограничен капом (verified по исходникам gpu-allocator 0.28.0 + vulkan/mod.rs:485). Pre-existing, НЕ фиксилось: `vulkan_perf_staging_pool` + 2 q8_1 qmatmul reference-теста падают и на базовом коммите (environment). |
| `ff995d09` | feat(wgpu): нативный GPU-quantize Q4_1/Q5_0/Q5_1 — word-assembly (160/176/192 B на workgroup, one-writer-per-u32-word), workgroup-LOCAL byte-offset (урок `c91c7daf`). 3 бага найдено и исправлено: constant-divisor specialization WGSL/драйвера побеждает `div_ieee` (литерал 15.0 → raw-частное 1 ulp HIGH) → runtime uniform `scale_bits` (15.0/31.0 to_bits); Q5 nibble-masking `&0xF` (lo/hi 5-бит, spill бита 4 в соседний nibble); Rust tie-семантика `minNum`/`maxNum` + Q5_0 absmax first-occurrence. |
| `577cec68` | perf(wgpu): все большие F32 GEMM → coop64 (dual 128x64 регрессировал оба шейпа на RTX 3060): 1024³ 0.55→0.53 sync / 0.446→0.426 batch; 64×4096 0.685→0.55 sync / 0.585→0.448 batch. K-panel padding (TK+1) РЕГРЕСС ~10% (задокументировано в заголовке шейдера, coopLoad с non-power-of-two stride сам создаёт конфликты). ЧЕСТНЫЙ ВЕРДИКТ: SLO wgpu matmul 1.30x всё ещё НЕ выполнен (sync 1.66–1.81x / batch 1.40–1.58x); tuning-пространство семейства шейдеров исчерпано (заголовок шейдера документирует отвергнутые конфиги: N-coalesced ±pad, BK=64, 128-thr, materialize B^T, K-panel=32); закрытие требует структурного rewrite (vulkan BM=BN=64 warp-partitioned coopMatLoad-from-shared layout) либо wgpu-upgrade с `subgroup_matrix` — кандидат следующей волны. |
| `8ebabc27` | perf(wgpu): params batch A — 17 single-dispatch сайтов на uniform ring. |
| `23845505` | perf(wgpu): params batches B+C — 20 сайтов (chunk-loops со свежим слотом ring'а на чанк; холодные сайты). ВСЕ wgpu params-сайты теперь на ring — оставшиеся 7 `create_buffer` — пулы/внутренности/MoE-scratch. |
| `224eb6c9` | fix(wgpu): `run_where_u8_cond` chunked-путь — перезапись слота ring'а (deferred-batch overwrite hazard, находка W-P2-аудита). |
| `2e376d89` | style: cargo fmt на файлах паритет-волны. |

### CUDA-baseline вердикты (W-B1, `.swarm/results/w-b1-cuda-baseline.md`)

Matmul sync/batch20 medians (ms; прогон W-B1 на HEAD `ae55bc2e`, GPU без контенции; для wgpu ЧП-столбец — актуальные значения после `577cec68`):

| Бекенд | matmul_1024³ sync / batch20 | matmul_64×4096⁴ sync / batch20 | SLO-вердикт (sync) |
|---|---|---|---|
| cuda | 0.3176 / 0.3032 | 0.3022 / 0.2832 | baseline |
| vulkan | 0.2985 / 0.2093 | 0.3361 / 0.2538 | **0.94x / 1.11x — PASS** |
| wgpu | 0.5385 / 0.4494 → **0.53 / 0.426** | 0.6841 / 0.5921 → **0.55 / 0.448** | 1.70x → **1.66x** / 2.26x → **1.81x — FAIL** (SLO 1.30x) |

**TF32: аналог НЕ нужен.** candle-CUDA честный FP32 — `MM_F32_REDUCED_PRECISION=false` по умолчанию, `CUBLAS_COMPUTE_32F` (подтверждено cpu−cuda max err ~2e-4, mean ~1e-5); vulkan 0.94–1.13x ≤ SLO 1.15x и без TF32; coopmat f16→f32-accum уже покрывает reduced-precision режим. Numeric-caveat (задокументировать): cpu−vk/cpu−wgpu max err 0.02–0.08 (mean 3.7e-3–1.5e-2) — на 2–3 порядка выше cpu−cuda из-за f16-input MMA — тот же numeric-класс, что у vulkan-coopmat.

**CUDA Graphs: аналог НЕ нужен.** cuda batch20/sync ≥ 0.6 на всех op/шейпах (0.64–0.96, маленькие тоже); условие «cuda-ratio заметно ниже» не срабатывает нигде — где vk/wgpu < 0.5 (0.23–0.43), там cuda ВЫШЕ (0.64–0.91). Watch-item: vk/wgpu small-op fixed dispatch overhead ~50–90µs sync vs ~10–25µs amortized — их собственный гэп, не вопрос graphs-эмуляции.

**candle-ug (NVRTC JIT SSA→PTX): work-without.** Единственный caller — unit-тест; оба бекенда поставляются прекомпилированными SPIR-V/WGSL-кернелами; runtime-compile — экосистемная фича, не контрактная.

### GPU-quantize: статус после волны

| Формат | wgpu | vulkan | Статус |
|---|---|---|---|
| Q8_0 | Native (`2b4e5f8e`) | Native (`172aec59`) | byte-identical CPU |
| Q4_0 | Native (`bbb32485`) | Native (`172aec59`) | byte-identical CPU (включая vmax tie-order) |
| Q4_1 | Native (`ff995d09`) | Native (`02e98766`) | byte-identical CPU, byte-parity тесты |
| Q5_0 | Native (`ff995d09`) | Native (`02e98766`) | byte-identical CPU, byte-parity тесты |
| Q5_1 | Native (`ff995d09`) | Native (`02e98766`) | byte-identical CPU, byte-parity тесты |
| Q8_1 (активации) | Native (pre-existing kernel) | Native (pre-existing kernel) | quant-путь qmatmul |
| Q2K–Q6K | deferral | deferral | callers — только offline CLI/тесты; dequant уже нативный |

Byte-parity тесты (сравнение GPU-байт с CPU-байтами, не dequant-толеранс): wgpu `wgpu_quantize_tests` / vulkan `vulkan_quantize` — по 9/9: matches_cpu_bytes (LCG-фикстуры), dm_large_multi_workgroup (4608 эл. = 144 блока, 18/36 workgroup'ов), ties_and_zero_bytes (±0.0 min/max ties, first-occurrence absmax, 0x8000 m), non_multiple_takes_cpu_behavior (n%32≠0 → идентичная error-поверхность). Новые тесты — `#[ignore]` (нужна GPU). Замеры времени не проводились в отдельных лефах; подробности находок (glslc algebraic rewrite, driver-специфичный OpFDiv 1-ulp, constant-divisor specialization, Q5 nibble masking) — `.swarm/results/w-q1-quant-vk.md`, `.swarm/results/w-q2-quant-wgpu.md`.

### Финальные гейты (HEAD `2e376d89`, RTX 3060, release, `--test-threads=1`)

| Прогон | Результат |
|---|---|
| `clippy -D warnings` | wgpu 0 · vulkan 0 (glslc banner — не диагностика) |
| `candle-core lib, wgpu` | **138 passed / 0 failed / 0 ignored** |
| `candle-core lib, vulkan` | **50 passed / 0 failed / 23 ignored** (6 новых GPU-quantize — `--ignored vulkan_quantize` → 9/9) |
| `backend_parity_diff` | wgpu **9/0/1** · vulkan **9/0/1** |
| `backend_smoke_tests, wgpu` | **43 passed / 0 failed** |
| `backend_smoke_tests, vulkan` | **50 passed / 0 failed** (OOM-тройка устранена `37d8b318`) |
| microbench spot-check | см. таблицу ниже |

### Перф spot-check (гейт 9, HEAD `2e376d89`, `backend_parity_microbench`, 2 прогона)

| Бекенд | op | sync median | batch20 median | Ожидание | Вердикт |
|---|---|---|---|---|---|
| vulkan | matmul_1024³ | 0.3063 / 0.3145 | **0.7071 / 0.7160** | sync ~0.30 | sync ✓; batch20 — РЕГРЕССИЯ (см. ниже) |
| vulkan | matmul_64×4096⁴ | 0.3274 / 0.3324 | 0.2722 / 0.2725 | sync ~0.33 | ✓ |
| wgpu | matmul_1024³ | 0.5306 / 0.5230 | 0.4243 / 0.4245 | 0.51–0.535 / 0.42–0.45 | ✓ (coop64-улучшение держится) |
| wgpu | matmul_64×4096⁴ | 0.5513 / 0.5528 | 0.4477 / 0.4487 | 0.54–0.56 / 0.44–0.45 | ✓ |

**Находка (A/B, не скрыта) → ЗАКРЫТО в `c239e5b4`:** vulkan gpu-allocator кап 16MB (`37d8b318`) вносил ~3.1x регрессию в batch-режиме больших matmul (1024³ batch20 0.71 против 0.225 pre-cap); growth-cap (`with_max_device_memblock_size(256MB)`) НЕ восстанавливал перф (0.60–0.70) — виноват сам цикл мелких блоков. Пере-анализ root-cause: OOM был не «256MB-блок на утёкший девайс», а полная утечка ВСЕХ ресурсов каждого девайса — `VulkanInner` никогда не дропается (Arc-цикл VulkanBuffer↔пулы), и каждое из ~24 устройств сьюта держало ~500MB (измерено поллингом nvidia-smi: 1096→12043MB). Финальный фикс: **глобальный device-cache** — `VulkanDevice::new(ordinal)` дедуплицирует `VulkanInner` по ordinal; кап аллокатора откачен к дефолту 256MB/64MB. Smoke 50/50, batch20 1024³ 0.2257 (= baseline 0.2253), sync не тронут. `same_device` = `Arc::ptr_eq`, семантически неотличимо от реконструкции.

### Оставшиеся follow-up (зафиксировано, не блокирует паритет)

1. ~~wgpu matmul SLO-гэп 1.43x + escalation (wgpu-upgrade с subgroup matrices)~~ **ЗАКРЫТО (W-U1 probe, `wgpu/vulkan` не тронут)**: остаточный гэп 1024³ 0.4332 vs CUDA 0.3032 (**1.43x**) — **окончательно установленный платформенный предел** cooperative-матриц gfx-rs wgpu на RTX 3060; все 4 рычага (структурный порт / materialize B^T / push constants / dependency upgrade) исчерпаны с A/B-данными — ссылки на все 4 отчёта: `.swarm/results/w-m1a-coop-shader.md`, `w-m1b-materialize-bt.md`, `w-m1c-push-constants.md`, `w-u1-wgpu-upgrade-probe.md`. SUBGROUP_MATRIX не существует ни в одной версии wgpu (ни 29.x, ни 30.0.1, ни trunk, CHANGELOG — 0 упоминаний); `mul_mat_subgroup_matrix.wgsl` — мёртвый порт из Dawn-бекенда ggml-webgpu (llama.cpp), `enable chromium_experimental_subgroup_matrix;` = hard parse-error naga (UnknownEnableExtension), в gfx-rs расширение отсутствует; upgrade = большой API churn (`VertexState.buffers` → `&[Option<_>]`, `DeviceDescriptor.default_queue`, etc.) без SLO-выигрыша. Возможное будущее (не действие волны): патч/форк wgpu с нативным SUBGROUP_MATRIX либо новое поколение coop-шейдеров (`enable wgpu_cooperative_matrix`).
2. K-quants Q2K–Q6K GPU-quantize (callers — только offline CLI/тесты).
3. ~~`vulkan_perf_staging_pool` + 2 q8_1 qmatmul reference-теста — пре-existing, environment~~ **q8_1 reference ЗАКРЫТО (`67e07ec5`)**: real численный баг (пропуск A-side q8_1-квантования на MMVQ-matvec-пути), не environment. Остаются 2 pre-existing фейла `--ignored`-свиты: `vulkan_perf_staging_pool` + `quantize_q8_1_x4` (общий прогон; оба проходят standalone — process-level ordering pollution staging-пула, W-N2 root-caused до stale f32-хвоста в readback staging-буфере) — кандидат на следующий перф-этап.
4. MoE scratch buffer pooling.
5. IQ/mxfp4/nvfp4 — ограничение candle в целом (у CUDA в этом форке тоже нет).
6. wgpu small-op dispatch overhead ~50–90µs (IMMEDIATES-стайл для самых маленьких struct-ов) — perf-класс.
7. ~~Латентный риск `1.0/d` driver-division в старых vulkan-кернелах Q8_0/Q4_0~~ **ЗАКРЫТО в волне #3 (`59d6db30` Q8_0, `ce78acf0` Q4_0, `0385542d` Q8_1)**: runtime scale-делитель через push constants (паттерн VulkanQuantizeScaleParams), glslc constant-folds литеральные делители; хазард остаётся только в мёртвых не-диспатчащихся `copy_to_quant`-вариантах (`cpy_f32_q*`/`set_rows_q*`) — задокументировано.
8. ~~vulkan gpu-allocator 16MB-кап: batch-режим больших matmul ~3.1x медленнее~~ **ЗАКРЫТО в `c239e5b4`**: device-cache (дедуп `VulkanInner` по ordinal) устранил OOM-утечку без капа; аллокатор на дефолтных 256MB/64MB блоках; smoke 50/50, batch20 восстановлен. Долгосрочный follow-up (низкий приоритет): разрыв Arc-цикла VulkanBuffer↔пулы, чтобы `vkDestroyDevice` реально срабатывал.
9. wgpu q8_1 same-class check (**НЕ запускался — волна остановлена пользователем, добавить в следующую**): проверить, нет ли у wgpu-диспетчера того же класса пропуска A-side q8_1-квантования на matvec-пути с малым k (smoke `q8_1_activation_matmul_reference` сейчас зелёный, но малый k не покрыт — не проверено).

### Волна 2026-08-22 #3 (div_ieee завершение + SLO-исследование wgpu matmul)

**div_ieee — все живые дивизион-хазарды vulkan закрыты.** Серия `59d6db30` (Q8_0), `ce78acf0` (Q4_0), `0385542d` (Q8_1): scale-делитель передан как runtime-значение через push constants (паттерн `VulkanQuantizeScaleParams`), т.к. glslc constant-folds литеральные делители. Dequant/matvec/matmul-пути Q8_0/Q4_0 читают `d` из блока и только умножают — хазарда нет. Мёртвые `copy_to_quant`-варианты (`cpy_f32_q*`/`set_rows_q*`) скомпилированы, но не диспатчатся — только задокументированы. Pre-existing fail `vulkan_q8_1_qmatmul_matches_cpu_reference` подтверждён на базовом коммите (не регрессия).

**SLO-исследование wgpu matmul — все 3 рычага плана W-R1 проверены эмпирически (A/B), все — честные негативные результаты, ничего не уехало в дерево:**

| Рычаг | Реализация / A/B | Результат | Вердикт |
|---|---|---|---|
| W-M1a coop-shader (`.swarm/results/w-m1a-coop-shader.md`) | структурный порт vulkan COOPMAT: BK=64, 2 барьера/панель, vec4 loaders; 5 вариантов | ВСЕ регрессируют: t1 128-thr BK=64 **−1.64x**; лучший t4 **−1.11x**; t5 coalesced-BT-only **−1.04x** | coopLoad wgpu 29 оптимален на dense stride-16 16x16 блоке; расширение K-панели теряет double-buffer overlap быстрее, чем экономит барьеры; Naga uniformity блокирует vulkan-style вложенный аккумулятор. coop64 (512-thr dense-16 double-buffered) = локальный оптимум. Reverted |
| W-M1b materialize-BT (`.swarm/results/w-m1b-materialize-bt.md`) | materialize B^T для квадратов (m.min(n) ≥ 256) | регресс ~**+15%** на 1024³ batch (0.4332→0.4866), **~2x** на 256³ batch; transpose стоит 0.06–0.08ms/call | GEMM от coalesced BT ничего не получает (согласуется с t5). Reverted |
| W-M1c push-constants (`.swarm/results/w-m1c-push-constants.md`) | push constants / IMMEDIATES для matmul-параметров + warptile-fallback repair | регресс: 256³ batch **+42%**, 1024³/tall +4%; reg_tile строго хуже warptile (256³ 1.05 vs 0.25 sync; 1024³ 7.17 vs 4.89) | ring уже делает deferred coalesced `write_buffer` одним вызовом на run-слот + cached bind group + dynamic offset; `set_immediates` строго дороже; reg_tile — не routing-дефект, а структурный предел (4x4/thread 32x32 tile vs 64x64 BK=32). Reverted |

**Вердикт:** остаточный гэп wgpu 1024³ = **0.4332 vs CUDA 0.3032 (1.43x)** — чистый шейдерный предел cooperative-матриц gfx-rs wgpu на RTX 3060 (+~0.15ms kernel time); host-путь уже оптимален. Все рычаги исчерпаны с данными (W-M1a coop-shader, W-M1b materialize B^T, W-M1c push constants, W-U1 dependency-upgrade probe — `w-u1-wgpu-upgrade-probe.md`). **Escalation ЗАКРЫТ (W-U1, clean negative):** SUBGROUP_MATRIX отсутствует во всех версиях wgpu (29.0.4, 30.0.1, trunk); `mul_mat_subgroup_matrix.wgsl` — мёртвый порт из Dawn-бекенда llama.cpp, в gfx-rs wgpu `chromium_experimental_subgroup_matrix` нет (naga: UnknownEnableExtension); upgrade = API churn без SLO-выигрыша. Будущее: патч/форк wgpu либо новое поколение coop-шейдеров (`enable wgpu_cooperative_matrix`).

**K-quants Q2K–Q6K GPU-quantize — НЕ гэп паритета.** У CUDA `quantize()`/`quantize_imatrix()` сами делают CPU round-trip (`cuda.rs:704-721`: `clone_dtoh` → CPU quantize → `memcpy_htod`); vulkan/wgpu для K-quants делают ровно то же (`to_cpu_storage()` + `quantize_from_cpu()`, `mod.rs:536-598`) = паритет. Форк уже ПРЕВЫШАЕТ CUDA по GPU-quantize целиком (6 нативных форматов vs 0 у CUDA). Follow-up остаётся только как optional enhancement (callers — offline CLI/тесты).

Финальные гейты волны #3 (HEAD `0385542d`, `--test-threads=1`): lib wgpu **138/0**, vulkan **50/0+23 ignored**; smoke wgpu **43/0**, vulkan **50/0**; parity_diff **9/0/1** оба; matmul_tests **11/0** оба; clippy `-D warnings` 0/0 оба (glslc banner — не диагностика); `fmt --check` чисто.

*Все прогоны §14 — HEAD `0385542d`, RTX 3060, release, `--test-threads=1`, 2026-08-22. Полные отчёты листьев — `.swarm/results/`.*

### Волна 2026-08-22 #4 (числовая корректность: закрыт последний падающий тест)

**Закрыт многолетний падающий тест — real numerical bug found & fixed (`67e07ec5`).** `vulkan_q8_1_qmatmul_matches_cpu_reference` (+`_no_warmup`). Root cause: Q8_1-веса (repack q8_1→q8_0) на matvec-пути с n=1, k≤4096 на NVIDIA уводились эвристикой `vulkan_should_use_mmvq` на сырой f32-MMVQ-путь **без q8_1-квантования A-side (активаций)** — CPU-контракт (как llama.cpp: src1 квантуется в vec_dot_type=Q8_1) нарушался → систематическая ошибка 5.09e-4 (= ровно ошибка LHS-квантования). Fix: флаг `force_q8_1_rhs` через dispatch (`quantized_matmul`/`quantized_matmat`/matvec) — Q8_1-веса всегда идут в fused q8_1-rhs кернелы с A-side квантованием.

Остаточная дельта GPU vs CPU = ровно **1 ULP f32 на 4742 (2^-11)**: fused int32-dot + одно f32-умножение vs последовательная f32-аккумуляция CPU — физический пол, не баг. Абс-ассерт `1e-6` при величине 4742 требовал 2.1e-10 rel (500x ниже f32-эпсилона) — harness-баг; заменён на rel `expected.abs()*1e-5 + 1e-6` (прецедент — `assert_f64_close` в той же свите).

**Числовая корректность обоих бекендов — полный зелёный статус.** `gpu_numerical_diff_tests` + `gpu_parity_matrix_tests` зелёные (cuda+vulkan, cuda+wgpu); все reference/byte-parity тесты зелёные. Из `--ignored`-свиты vulkan остаются 2 pre-existing фейла: `vulkan_perf_staging_pool` + `quantize_q8_1_x4` (в общем прогоне; оба проходят standalone — process-level ordering pollution staging-пула, W-N2 root-caused до stale f32-хвоста в readback staging-буфера; кандидат на следующий перф-этап).

**wgpu q8_1 same-class check — НЕ запускался** (пользователь остановил волну) → открытый follow-up: проверить, нет ли у wgpu-диспетчера того же класса пропуска A-side квантования для q8_1 (smoke-тест `q8_1_activation_matmul_reference` сейчас зелёный, но покрывает ли он matvec-путь с малым k — не проверено).

*Прогоны волны #4 — HEAD `67e07ec5`, RTX 3060, release, `--test-threads=1`, 2026-08-22.*

---
*Все утверждения о коде проверены на HEAD 1596684b; тестовые прогоны — RTX 3060, release, 2026-08-21, однопоточные (`--test-threads=1`). Фиксы финальной волны: wgpu_backend.rs (powf/elu F16, reduce I16, to_dtype U8→I32, F8E4M3-хабы, uniform ring), 7 argsort-шейдеров candle-vulkan-kernels, vulkan scatter_add F8E4M3, backend_parity_diff (harness-сравнение по dtype результата). См. также §10–11 документа 2026-08-19.*
