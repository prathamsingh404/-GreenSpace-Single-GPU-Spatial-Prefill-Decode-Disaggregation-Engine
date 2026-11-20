# GreenSpace audit · 21 November 2026

**Disposition.** Physical SM partitioning works on the tested RTX 5050 Laptop GPU after correcting resource creation. Decode variance improved once and regressed on repetition. The strict Checkpoint 1 gate remains open. The repository has no verified end-to-end serving speedup or established 25 ms token-latency guarantee.

## Reproduced defects

| Finding in the original checkout | Change and evidence |
| --- | --- |
| Both Green Contexts received the full device descriptor. The original engine reported 8 and 12 SMs. A driver query on the two actual handles returned **20 and 20**. All 19 original tests passed on that checkout. | The normal engine entry point now creates both streams from one split and uses the returned counts. The physical-ID kernel observed all 20 SMs across two disjoint sets in both split orientations. |
| The decode pool sent completed prefills back through prefill when full. Reallocation could replace a live block table. | Admission applies backpressure and completed handoffs retain their pages. The regression submitted 64 requests with one available decode slot. Every request produced its exact token count with one prefill per request and all pages returned. |
| A request with `max_tokens=1` returned two tokens. A decode allocation failure could terminate the worker. A prefill error could leave a caller polling indefinitely. | Both engines honor first-token completion. Worker errors produce a terminal failed request. A one-block test forces decode exhaustion, checks the error and completes the next request on the same worker. |
| Rebinding destroyed live contexts. CTX initialization could silently select an ordinary priority stream. Shutdown could release contexts after a two-second join timeout. | Spatial initialization propagates driver failures. Pair selection reuses owned streams. In-place SM mutation is rejected. Shutdown joins workers before closing contexts and cancels unfinished requests. |
| The barrier diagnostic initialized buffers on a third stream without ordering those writes before the test. The web endpoint could return success even when the GPU check failed to run. | Inputs are initialized on the producer stream. The web smoke test obtained a successful hardware check while the engine was running and HTTP 503 after shutdown. Both web modes returned exactly one requested token. |
| Decode gathered pages using advanced indexing. Charts and UI telemetry presented hard-coded or random data as hardware evidence. One ablation hard-coded a one-second denominator and disabling its controller missed the reference held by the worker. | Documentation distinguishes metadata handoff from device-side gathering. Charts and API samples carry simulation labels. The harness measures the elapsed interval, rejects failed/incomplete requests and disables the worker's controller for the fixed configuration. |

## Hardware measurements

The GPU reported compute capability 12.0, 20 SMs, a 32 MiB L2 cache and an eight-SM structured split quantum. The run used Windows WDDM driver 621.24, CUDA Driver API 13.5 and PyTorch 2.13.0+cu130. The device resource supplied no valid locality-domain flag. Each physical test launched 1,024 blocks **per stream**. Each concurrency test timed 20 GEMVs per condition alongside 4,096-square FP16 GEMMs. Variance is sample variance in ms². Overlap is the sum of intersecting kernel intervals in the profiled spatial run.

| Measurement | Initial run | Full-suite repetition |
| --- | ---: | ---: |
| SM IDs shared between phases, both pairs | 0 | 0 |
| Unpartitioned decode variance | 0.000413 | 0.000395 |
| Spatial decode variance | 0.000089 | 0.047968 |
| Concurrent GPU kernel time | 34.263 ms | 33.822 ms |
| PyTorch allocated-byte change over five lattice lifecycles | 0 | 0 |
| Strict hardware gate | Pass | **Fail** |

The second spatial sample set included a 1.169 ms decode, with median 0.211 ms. Fixed sampling order, twenty observations per mode and an active display GPU limit causal attribution. Both traces contained twenty kernels on each of two CUDA streams. Teardown checked driver destroy results. PyTorch allocator accounting measures tensor allocations and does not count driver-internal resources. [First run](../artifacts/checkpoint_1_first_run.json) · [Full-suite run](../artifacts/checkpoint_1_report.json) · [Compressed Chrome trace](../artifacts/checkpoint_1_overlap_trace.json.gz)

A separate run of the existing interference harness measured mean host-observed reduction latencies of 0.094 ms in isolation, 0.129 ms alongside GEMMs and 0.159 ms with a lower creation-time prefill workqueue limit. This comparison supplies no evidence of an improvement from lowering the limit on this workload.

## Final verification

The standalone C++ diagnostic built with CMake, Clang, NVIDIA CUDA headers and NVRTC. It independently observed prefill IDs 2, 3, 6, 7, 10, 11, 14, 15, 16, 17, 18 and 19, and decode IDs 0, 1, 4, 5, 8, 9, 12 and 13. The packaged Python engine now runs the physical-ID check before serving. The wheel build passed with Python warnings treated as errors and contains the CUDA kernel source and dashboard HTML.

The final full suite passed 23 of 24 tests. The sole failure was the hardware variance comparison. The latest measured variance was 0.0000983 ms� on ordinary streams and 0.0002702 ms� on spatial streams, with 2.821 ms of verified kernel overlap. In six additional paired trials with preinitialized timing events and alternating order, spatial variance was lower in five. No failed run was discarded. [Final test outcomes](../artifacts/final_verification/test_summary.json) � [Final hardware report](../artifacts/final_verification/checkpoint_1_report.json) � [Six paired trials](../artifacts/checkpoint_1_variance_trials.json)

The existing burst harness completed seven synthetic-weight requests per engine, with 125 tokens and 118 inter-token intervals each. Measured P99 was 638.75 ms on the monolithic engine and 243.05 ms on the spatial engine. Both exceeded 25 ms. The two-request ablation also ran in all four of its existing modes. Its controller-observation configuration uses a fixed split throughout. The proposed chunked and dynamically actuated serving comparisons remain outstanding. [Burst samples](../artifacts/synthetic_burst_smoke.json) � [Ablation smoke run](../artifacts/synthetic_ablation_smoke.json)

## Coverage against the brief

| Checkpoint | Evidence and outstanding acceptance work |
| --- | --- |
| 1 · Hardware | Native driver resource query, complementary split, SM-ID sampling, startup lattice, stream reuse, separate full-device stream and real-device tests. Kernel overlap reproduced. Lower variance did not reproduce. Engine startup now runs the same physical-ID diagnostic on each spatial pair before allocating the model. |
| 2 · KV ownership | Shared tensors, event handoff and allocation-error recovery exist. Per-page state/reference counts, transfer/allocation tracing over 100 handoffs, and a 500-iteration tearing test with a negative control remain open. |
| 3 · Numerical equivalence | Shape tests and synthetic-model inference pass. The runner lacks pretrained weights, RoPE and position-aware chunked prefill. No 64-step reference-model equivalence result exists. |
| 4 · Runtime | Real dual workers, successful small-model admission stress and post-OOM recovery. Per-request Python decode loops and host-side token sampling remain. The 64-request long-context stress distribution, packed decode and sustained engine-trace overlap threshold are unverified. |
| 5 · Controller | The controller calculates proposed workqueue and chunk-size changes. The prefill path still processes whole prompts on a fixed pair. The calibrated roofline, safe cross-pair runtime transitions, effective budget actuation, and hardware shock/low-load tests remain open. |
| 6 · Profiling | The hardware microbenchmark produced a Kineto GPU trace. The repository lacks measured before/after Nsight Compute hardware counters for decode memory stalls during engine operation. |
| 7 · Serving evaluation | The legacy harnesses can exercise synthetic workloads. The proposed four-way traced ablation, per-request SLO goodput, independent runs and measured plots remain open. The historical P99 improvement is unsupported. |

The package keeps the existing `greenspace/` import tree. The CP1 hardware modules live under `greenspace/hardware/` so tests and the public engine use the same implementation. Commands and toolchain requirements are in the [README](../README.md).

## Research comparison

- **DistServe** separates phases across GPU resources and measures goodput under first-token and per-output-token objectives. **Mooncake** also uses a distributed KV cache across the memory and storage hierarchy. Their results motivate measuring the local handoff and both SLOs. They do not establish a fixed network transfer penalty for all distributed configurations. [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf) · [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin)
- **Sarathi-Serve** chunks prefills and schedules them with ongoing decodes. A comparable baseline needs real position-correct chunk execution, matched arrivals and measured inter-token latency. **PagedAttention** manages logical KV blocks through physical page mappings. Sharing a block table does not make advanced-index gathers free. [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf) · [PagedAttention](https://arxiv.org/abs/2309.06180)
- **DuetServe** already combines single-GPU spatial/temporal sharing, a roofline model, SM-split selection and asynchronous dispatch. **AgentServe** already describes pre-established Green Context slots, TPOT-driven adaptation and dynamic prefill budgeting on a consumer GPU. The broad novelty claims overlap directly with prior work. A narrower claim about the control law would require a concrete implementation and ablation against these approaches. [DuetServe §4](https://arxiv.org/html/2511.04791v1) · [AgentServe §III](https://arxiv.org/html/2603.10342v1)
- **NVIDIA** documents split alignment, remainder semantics and shared workqueue behavior. Disjoint SM sets provide no guarantee of simultaneous execution or forward progress. The isolation probe and the timeline therefore answer separate questions. [CUDA Green Contexts](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GREEN__CONTEXTS.html)
