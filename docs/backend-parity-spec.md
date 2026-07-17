# Backend Parity Specification

## 1. Назначение

Этот документ задаёт проверяемые критерии функционального, численного и производительного паритета Vulkan- и WebGPU-бэкендов Candle относительно CUDA baseline.

Он не фиксирует конкретную внутреннюю архитектуру до аудита репозитория. Реализация должна использовать существующие abstractions Candle, если их расширение достаточно.

## 2. Scope и baseline

В начале проекта зафиксировать:

- repository path;
- branch;
- baseline commit;
- dirty working tree;
- Rust toolchain;
- Cargo features;
- CUDA toolkit/runtime;
- Vulkan loader/runtime;
- `wgpu` и `naga` versions;
- OS;
- CPU;
- GPU;
- driver;
- доступные device features и limits.

Baseline паритета — фактическая backend-independent поверхность CUDA-бэкенда зафиксированного commit.

В scope входят все операции, которые пользователь Candle может вызвать через общий tensor/backend API и которые фактически исполняются CUDA.

Не включаются автоматически:

- NCCL и другие CUDA-specific distributed integrations;
- отладочные internal helpers без backend-independent семантики;
- vendor API, для которого Candle не предоставляет общий backend contract.

Любое исключение должно быть явно отражено как `CudaSpecific` с обоснованием.

## 3. Backend profiles

### 3.1 Native Vulkan

Цель: максимально полный native GPU parity с CUDA на Vulkan-capable hardware.

Допустимые механизмы:

- SPIR-V;
- Vulkan extensions;
- specialization constants;
- subgroup operations;
- cooperative matrices;
- timestamp queries;
- pipeline cache;
- vendor-specific fast paths;
- architecture-specific autotuning.

Условия:

- runtime capability detection;
- generic GPU fallback;
- отсутствие CPU computation fallback;
- отдельное тестирование fast и generic paths.

### 3.2 Native WebGPU

Цель: максимальный parity через нативный `wgpu` на Vulkan/DX12/Metal.

Native-only `wgpu` features допустимы, если:

- feature обнаруживается через adapter/device capabilities;
- path не считается portable;
- существует корректный GPU path без feature либо явная typed unsupported error;
- backend differences протестированы там, где доступно оборудование.

### 3.3 Portable WebGPU

Цель: переносимый WebGPU/WGSL path для browser/WASM.

Ограничения portable WGSL должны быть отражены в capability matrix. Нельзя использовать native-only `wgpu` extensions и затем заявлять browser compatibility.

Для dtype или операции, отсутствующих в portable profile, допустимы:

1. корректная GPU-эмуляция;
2. явная unsupported capability;
3. backend-specific API limitation, видимая пользователю.

Silent cast запрещён.

## 4. Parity manifest

Создать machine-readable manifest в формате, естественном для репозитория, например:

- `backend-parity.toml`;
- `backend-parity.json`;
- Rust data table, из которой генерируется документация.

Manifest должен быть проверяемым CI.

### 4.1 Обязательные поля записи

Для каждой operation/dtype комбинации:

- operation identifier;
- public/backend trait entry point;
- semantic description;
- CUDA source;
- Vulkan source;
- native WebGPU source;
- portable WebGPU source;
- profile status;
- storage dtype;
- compute dtype;
- accumulator dtype;
- supported ranks;
- shape restrictions;
- contiguous support;
- strided support;
- broadcasting;
- scalar support;
- zero-sized support;
- views/slices;
- transpose/permutation;
- deterministic status;
- forward coverage;
- backward/autodiff coverage;
- correctness test;
- differential test;
- property/random test;
- benchmark;
- capability requirements;
- known limitations;
- evidence/source link;
- owner or tracking issue.

### 4.2 Статусы

`Native`  
Операция исполняется нативным GPU kernel и прошла базовую проверку.

`Optimized`  
Есть специализированный fast path и benchmark, подтверждающий его использование/эффект.

`GPUEmulated`  
Семантика реализована на GPU через составные операции или программную эмуляцию отсутствующего типа.

`UnsupportedBySpecification`  
Целевой стандарт не предоставляет нужной возможности; есть ссылка на нормативный источник и оценка эмуляции.

`UnsupportedByHardware`  
Текущий adapter не предоставляет обязательную capability; приложен capability dump.

`CudaSpecific`  
Нет backend-independent эквивалента в Candle API.

`Missing`  
Работа не реализована.

`Verified`  
Операция прошла обязательный correctness suite для данного профиля.

Статус `Optimized` не заменяет `Verified`: в manifest могут использоваться отдельные поля implementation/verification.

### 4.3 CI-инварианты

CI должен обнаруживать:

- новую CUDA operation без manifest entry;
- отсутствующий профиль;
- `Verified` без test reference;
- `Optimized` без benchmark reference;
- `GPUEmulated` без явной capability/reporting semantics;
- неизвестный статус;
- `Missing` в release-required scope;
- documentation, не совпадающую с machine-readable manifest.

## 5. Operation inventory

Фактический список строится из CUDA source и backend traits. Минимально проверить следующие семейства.

### 5.1 Device и storage

- device creation;
- capability query;
- allocation;
- deallocation;
- buffer reuse/pooling;
- host-to-device;
- device-to-host;
- device-to-device;
- copy with offsets/layouts;
- synchronization;
- device loss;
- error propagation;
- pipeline/shader cache;
- resource cleanup.

### 5.2 Tensor construction и layout

- zeros;
- ones;
- full;
- arange;
- random initialization;
- contiguous;
- reshape;
- narrow/slice;
- views;
- transpose;
- permute;
- broadcasting;
- copy across non-contiguous layouts.

### 5.3 Elementwise и conversion

- unary arithmetic;
- transcendental functions;
- binary arithmetic;
- comparisons;
- min/max/clamp;
- conditional/where;
- dtype conversion;
- bit operations where supported.

### 5.4 Reductions

- sum;
- mean;
- min/max;
- argmin/argmax;
- variance-related primitives;
- norm;
- softmax reductions;
- arbitrary axes;
- keepdim behavior;
- empty dimensions and error semantics.

### 5.5 Indexing

- index-select;
- gather;
- scatter;
- embedding;
- masking;
- sorting;
- top-k;
- duplicate indices;
- out-of-range errors.

### 5.6 Linear algebra

- matmul;
- batched matmul;
- transposed operands;
- non-contiguous operands;
- mixed precision;
- quantized matmul;
- small latency-sensitive shapes;
- large throughput shapes.

### 5.7 Neural-network operations

- convolution;
- transposed convolution;
- pooling;
- normalization;
- LayerNorm;
- RMSNorm;
- activation functions;
- softmax/log-softmax;
- RoPE;
- attention helpers;
- causal masking;
- model-specific fused operations.

### 5.8 Quantization

- supported low-bit storage formats;
- dequantization;
- quantized matmul;
- scale/zero-point semantics;
- packed layout;
- tail handling;
- accumulator precision.

### 5.9 Autodiff и custom operations

- backward paths used by common operations;
- gradient accumulation;
- custom op contract;
- error propagation;
- supported layout/dtype combinations.

## 6. Functional acceptance

Операция считается функционально закрытой для профиля, если:

1. Реализована заявленная семантика.
2. Не используется CPU computation fallback.
3. Все поддерживаемые dtype/layout варианты задокументированы.
4. Unsupported combinations возвращают явную ошибку.
5. Есть correctness tests.
6. Пройдены edge cases.
7. Для `GPUEmulated` измерена стоимость.
8. Manifest обновлён.
9. Нет validation errors и resource leaks.

Компиляция без исполнения не считается закрытием.

## 7. Numerical validation

### 7.1 Reference chain

Использовать:

`high-precision CPU reference ↔ CUDA ↔ Vulkan ↔ native WebGPU ↔ portable WebGPU`

CPU reference должен использовать наиболее точную разумную реализацию, а не повторять GPU-алгоритм с теми же ошибками.

### 7.2 Discrete outputs

Точное совпадение требуется для:

- integer arithmetic, если семантика не допускает overflow variation;
- bool;
- comparisons;
- shapes;
- indices;
- sorting/top-k ordering с заранее определённой tie semantics;
- deterministic random sequences, если exact sequence является частью API.

### 7.3 Floating point

Для каждой operation/dtype определить:

- absolute tolerance;
- relative tolerance;
- ULP threshold, если применимо;
- accumulator dtype;
- NaN/Inf policy;
- denormal policy;
- determinism expectation.

Процедура:

1. Измерить CUDA против CPU reference.
2. Измерить Vulkan/WebGPU против CPU reference.
3. Сравнить распределение ошибок, а не только максимум.
4. Не ослаблять tolerance автоматически.
5. Любое исключение документировать по operation/dtype/hardware.

### 7.4 Обязательные формы данных

- scalar;
- zero-sized;
- one-element;
- odd dimensions;
- prime dimensions;
- non-power-of-two;
- large dimensions near limits;
- contiguous;
- strided;
- slices with offsets;
- transposed;
- broadcast;
- misalignment cases, допустимые API;
- repeated/aliased views, если API их допускает.

### 7.5 Обязательные значения

- zero;
- signed zero;
- one;
- negative values;
- smallest/largest normal;
- subnormal;
- NaN;
- positive/negative Inf;
- overflow-prone;
- underflow-prone;
- division by zero;
- duplicate indices;
- invalid indices and parameters.

### 7.6 Randomized/property testing

- фиксированные seeds;
- сохранение failing seed;
- shrink/minimization, если доступно;
- сравнение нескольких ranks, layouts и dtype;
- отдельный nightly/extended suite для дорогих cases.

## 8. Performance specification

### 8.1 Принцип

Производительность оценивается на трёх уровнях:

1. microkernels;
2. composed GPU workloads;
3. end-to-end model workloads.

Release decision основывается прежде всего на end-to-end и hot paths.

### 8.2 Native Vulkan SLO

На одном GPU с CUDA baseline:

- end-to-end median: не более 15% медленнее CUDA;
- hot-path geometric mean: не более 20% медленнее CUDA;
- critical kernel: не более 1,5× медленнее без документированного ограничения;
- 10% от CUDA — stretch goal.

### 8.3 Native WebGPU SLO

- end-to-end median: не более 30% медленнее CUDA;
- critical kernel: не более 2× медленнее без профиля и объяснения;
- 20% от CUDA — stretch goal;
- 10% — stretch goal только для отдельных kernels.

### 8.4 Portable WebGPU

Универсальный CUDA percentage не применяется.

Обязательные показатели:

- browser/OS/adapter;
- native WebGPU comparison на том же GPU, если возможно;
- time to first usable pipeline;
- steady-state latency;
- throughput;
- memory consumption;
- отсутствие CPU fallback/readback;
- investigation для hot path >2× native WebGPU.

### 8.5 Benchmark environment

Фиксировать:

- commit;
- build profile;
- compiler flags;
- Cargo features;
- OS;
- CPU;
- GPU;
- driver;
- CUDA/Vulkan/WebGPU versions;
- browser;
- power state;
- GPU temperature;
- available VRAM;
- background load.

### 8.6 Benchmark protocol

- release build;
- одинаковые inputs;
- одинаковые dtype;
- одинаковые shapes;
- одинаковые layouts;
- одинаковая математическая семантика;
- warm-up;
- явная GPU synchronization;
- достаточное число repetitions;
- median;
- p90/p95;
- variance;
- cold start отдельно;
- shader compilation отдельно;
- pipeline creation отдельно;
- allocation отдельно;
- transfer отдельно;
- dispatch отдельно;
- kernel execution отдельно;
- end-to-end отдельно.

Измеряемый участок не должен содержать случайный readback или host-side preprocessing, отсутствующий у baseline.

### 8.7 Минимальный benchmark suite

Microbenchmarks:

- unary;
- binary;
- broadcast;
- cast;
- reduction;
- softmax;
- LayerNorm/RMSNorm;
- gather/scatter;
- embedding;
- matmul;
- batched matmul;
- convolution;
- quantized matmul;
- dequantization;
- RoPE;
- masking;
- transpose/layout conversion.

End-to-end:

- одна небольшая transformer model из существующих examples;
- одна representative LLM workload;
- prefill;
- decode;
- time to first token;
- tokens per second;
- batch size 1;
- throughput-oriented batch;
- convolution workload, если соответствующий example доступен.

## 9. Profiling и optimization

Оптимизировать только после доказанного bottleneck.

Проверить:

- workgroup size;
- subgroup size;
- cooperative matrix availability;
- vectorized loads/stores;
- memory coalescing;
- workgroup/shared memory;
- bank conflicts;
- register pressure;
- occupancy;
- divergence;
- reduction strategy;
- accumulator precision;
- dispatch count;
- command buffer batching;
- bind group/descriptor reuse;
- buffer pools;
- pipeline cache;
- specialization;
- fusion;
- layout-specialized kernels;
- dtype-specialized kernels;
- autotuning;
- barriers;
- copies;
- host synchronization;
- readback.

Vendor-specific optimization обязана сохранять generic GPU path.

## 10. Implementation workflow

### Phase 0 — Safety

- прочитать repository rules;
- проверить Git;
- сохранить пользовательские изменения;
- создать branch/worktree;
- зафиксировать environment.

### Phase 1 — Baseline

- собрать workspace;
- выполнить существующие tests;
- выполнить доступные backend tests;
- выполнить baseline benchmarks;
- сохранить failures и raw results.

### Phase 2 — Audit

Поискать:

- `todo!`;
- `unimplemented!`;
- `panic!`;
- `TODO`;
- `FIXME`;
- `unsupported`;
- `fallback`;
- CPU conversions;
- readback;
- `map_async`;
- forced synchronization;
- missing feature branches;
- shader stubs.

Создать первый parity manifest.

### Phase 3 — Core runtime

Сначала закрыть:

- device/capability abstraction;
- allocation/storage;
- transfers;
- synchronization;
- shader compilation;
- pipeline caching;
- error handling;
- resource lifetime.

### Phase 4 — Vertical operation slices

Для каждой operation:

1. определить semantics;
2. реализовать kernel;
3. реализовать dispatch;
4. покрыть dtype/layout;
5. добавить correctness tests;
6. добавить differential tests;
7. добавить benchmark;
8. профилировать;
9. оптимизировать;
10. обновить manifest/docs.

### Phase 5 — End-to-end

- запустить model workloads;
- определить top bottlenecks;
- оптимизировать по профилю;
- проверить regressions;
- измерить SLO.

### Phase 6 — Independent review

Независимый reviewer проверяет:

- полноту manifest;
- отсутствие CPU fallback;
- silent casts;
- numerical tolerances;
- benchmark validity;
- unsafe code;
- resource lifetime;
- API compatibility;
- unsupported classifications;
- финальные заявления.

## 11. CI requirements

Добавить применимые проверки:

- formatting;
- linting;
- workspace build;
- unit/integration/doctests;
- Vulkan build/tests;
- native WebGPU build/tests;
- WASM/portable WebGPU build;
- shader validation;
- parity manifest validation;
- differential test smoke suite;
- benchmark smoke suite;
- feature matrix checks.

Hardware skip должен сообщать:

- какое устройство искалось;
- какая capability отсутствует;
- что именно не было проверено.

Skip не считается успешной верификацией.

## 12. Release acceptance

Release-ready состояние достигнуто, когда:

1. Вся CUDA backend-independent surface отражена в manifest.
2. Нет неизвестных статусов.
3. Нет `Missing` в обязательном scope.
4. Unsupported status подтверждён доказательством.
5. Нет скрытых CPU fallback.
6. Нет silent dtype/precision conversion.
7. Все реализованные operations имеют tests.
8. Critical operations имеют differential tests.
9. Hot paths имеют benchmarks.
10. Vulkan выполняет release SLO либо отклонение отдельно согласовано.
11. Native WebGPU выполняет release SLO либо отклонение отдельно согласовано.
12. Portable WebGPU имеет отдельную capability/performance matrix.
13. Проходят применимые CI checks.
14. Нет validation errors и известных resource leaks.
15. Нет незакрытых critical review findings.
16. Документация соответствует коду и manifest.

Stretch goals не являются release blockers.

## 13. Финальный отчёт

Отчёт должен включать:

- baseline/final commit;
- branch;
- environment;
- architecture changes;
- changed files;
- capability matrix;
- operation/dtype/layout matrix;
- test commands и результаты;
- numerical error tables;
- benchmark methodology;
- raw benchmark artifacts;
- CUDA vs Vulkan;
- CUDA vs native WebGPU;
- native vs portable WebGPU;
- end-to-end model results;
- memory usage;
- cold/steady-state results;
- найденные и удалённые fallback;
- ограничения и доказательства;
- использованные референсы;
- заимствованный код и лицензии;
- independent review findings.

Только фактические результаты. Никаких заявлений о выполнении без логов и измерений.

## 14. Нормативные и первичные источники

При актуализации спецификации использовать прежде всего:

- xAI Grok Build project rules: `https://docs.x.ai/build/features/project-rules`
- xAI Grok Build Plan mode: `https://docs.x.ai/build/modes-and-commands`
- Candle upstream: `https://github.com/huggingface/candle`
- WGSL specification: `https://www.w3.org/TR/WGSL/`
- `wgpu` feature reference: `https://docs.rs/wgpu-types/latest/wgpu_types/struct.Features.html`
- Vulkan cooperative matrix reference: `https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html`
- NVIDIA cuBLAS/cuBLASLt documentation: `https://docs.nvidia.com/cuda/cublas/`
- NVIDIA cuDNN Backend documentation: `https://docs.nvidia.com/deeplearning/cudnn/backend/latest/index.html`

Версии источников и даты проверки фиксировать в финальном отчёте.
