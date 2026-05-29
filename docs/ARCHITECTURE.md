# Project GreenSpace Architecture

Initial research and systems design notes.
Prefill-Decode Disaggregation on Single GPUs.

## Memory Bandwidth Contention in Shared L2 Caches

Prefill GEMM kernels saturate L2 read ports and interfere with Decode.
