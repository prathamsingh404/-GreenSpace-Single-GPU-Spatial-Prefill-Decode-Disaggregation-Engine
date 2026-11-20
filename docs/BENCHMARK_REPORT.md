# Benchmark evidence

The historical 480.0 ms and 29.8 ms P99 values came from the values in `benchmarks/plot_results.py`. The plotting code also inserts example burst spikes and synthesizes jitter. The resulting 93.8% reduction and the bundled plots cannot establish measured serving performance. A P99 of 29.8 ms would exceed the proposed 25 ms target.

The dashboard generates its latency, utilization and attention animations from fixed values and random samples. The barrier endpoint and the hardware checkpoint exercise the physical GPU. The GPU overlap test uses GEMM and GEMV microbenchmarks. Its times apply to that workload and hardware.

The burst and ablation scripts are experimental harnesses. The prototype lacks chunked prefill and a runtime actuator for the proposed split/budget changes. Publication-quality ablations still require identical arrival traces, calibrated measurements, a real model, completed-request checks, per-request SLO goodput and independent repetitions. [Audit](AUDIT.md) · [Hardware measurements](../artifacts/checkpoint_1_report.json)
