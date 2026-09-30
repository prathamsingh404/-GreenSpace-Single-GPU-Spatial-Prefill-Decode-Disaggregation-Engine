# Project GreenSpace Architecture

Initial research and systems design notes.
Prefill-Decode Disaggregation on Single GPUs.

## Memory Bandwidth Contention in Shared L2 Caches

Prefill GEMM kernels saturate L2 read ports and interfere with Decode.

```
cuDeviceGetDevResource -> cuDevResourceGenerateDesc -> cuGreenCtxCreate -> cuCtxFromGreenCtx
```

## Green Context vs CUDA MPS Architectural Comparison

Unlike MPS, Green Contexts require zero root daemons and allow dynamic in-process partition rebinding.

## Zero-Copy Pointer Flipping Architecture

HBM allows cross-stream memory sharing with zero byte transfers.

## Dual Queue Scheduling Logic

Independent threads ingest prompts and generate tokens concurrently.
