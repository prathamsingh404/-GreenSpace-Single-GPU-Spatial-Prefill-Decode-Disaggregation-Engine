# CUDA Green Contexts in GreenSpace

The driver wrapper uses NVIDIA's typed `cuda.bindings.driver` structures. It queries the device SM resource and passes it to `cuDevSmResourceSplitByCount`. One output group and its remainder supply both halves of a pair. Separate splits on the same device can select the same physical SMs.

For each half, GreenSpace generates a descriptor, creates a Green Context and binds a nonblocking stream with `cuGreenCtxStreamCreate(stream, context, flags, priority)` in the C API. PyTorch uses the stream through `ExternalStream`. The owner synchronizes its streams before destruction. New SM or workqueue allocations require a different context. The current worker streams are fixed for an engine lifetime.

The minimum size and alignment depend on the device and split flags. The observed device has 20 SMs, reports an eight-SM minimum and alignment, and returns a 12-SM remainder for an eight-SM group. The manifest records a locality ID only if the resource flags identify a locality domain. Driver version and symbol availability alone cannot prove isolation. Distinct SM masks also leave memory bandwidth shared and do not guarantee concurrent execution or forward progress. [NVIDIA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GREEN__CONTEXTS.html)

If the driver rejects a resource configuration, inspect the queried size, quantum, returned group count and remainder. If a required entry point is missing, match the driver, bindings and supported toolchain, or select unpartitioned execution explicitly. If the profiler records no CUDA kernels, check that the PyTorch build includes CUPTI and that profiling is enabled. Empty traces fail the overlap gate.
