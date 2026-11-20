# Project GreenSpace

GreenSpace is a research prototype for running LLM prefill and decode on complementary groups of streaming multiprocessors (SMs) on one NVIDIA GPU. Two CPU workers share model weights and a device-resident KV cache. The included transformer uses synthetic weights and greedy decoding.

The hardware test recorded disjoint physical SM IDs on an RTX 5050 Laptop GPU. The test covers a 12-SM prefill / 8-SM decode pair and its reverse, and measures kernel overlap independently. Request-level latency and SLO goodput still need a controlled serving benchmark. [Hardware manifest](artifacts/checkpoint_1_hardware.json) · [Latest hardware test results](artifacts/final_verification/checkpoint_1_report.json) · [Project audit](docs/AUDIT.md)

## Install and verify

Use a Python environment with a CUDA-enabled PyTorch build supporting your GPU's compute capability. This checkout was tested on Windows with Python 3.13, PyTorch 2.13.0+cu130, cuda-bindings 13.5 and NVIDIA driver 621.24. The native diagnostic also requires the NVIDIA driver import library, CUDA headers, NVRTC and a 64-bit C++ compiler.

```powershell
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[audit,web]"
python -m greenspace.hardware.topology_probe
python -m pytest -q -W error
python -m benchmarks.check_partition_variance
```

To run the standalone CUDA probe with a toolkit installation visible to CMake, use the following commands. A multi-configuration generator places the executable under `build/Release`. A single-configuration generator places it under `build`.

```powershell
cmake -S csrc -B build
cmake --build build --config Release
.\build\Release\green_ctx_probe.exe
```

## Run the prototype

```python
import time
from greenspace import GreenSpaceEngineCore

with GreenSpaceEngineCore() as engine:
    request = engine.add_request([101, 205, 309, 412], max_tokens=8)
    while not request.is_finished():
        time.sleep(0.01)
    if request.error:
        raise RuntimeError(request.error)
    print(request.generated_token_ids)
```

The engine reports the SM counts returned by the driver. Spatial mode requires successful resource splitting, context creation and stream creation. Setting `config.greenspace.enable_green_contexts = False` explicitly selects ordinary streams on the shared GPU. Workqueue resources are configured at context creation. The current controller records proposed changes during inference.

Run `python run_web.py` for the local demonstration. The dashboard animates simulated utilization, attention and latency. Its generation endpoint exercises the synthetic-weight model using token IDs derived from the requested length. [Benchmark provenance](docs/BENCHMARK_REPORT.md)

## Implementation

- [Hardware and native CUDA bindings](greenspace/hardware/green_context_pool.py) provision complementary pairs from the same driver split and reuse their streams.
- [Engine](greenspace/core/engine.py) manages the workers, the shared model and request cleanup. [Block manager](greenspace/core/block_manager.py) tracks physical page ownership per request.
- [Attention](greenspace/models/attention.py) writes prompt and decode KV into the shared pool and gathers pages for attention. The handoff passes metadata and a CUDA event.
- [Verification](tests/test_cp1_hardware_isolation.py) launches 1,024 blocks per stream, reads the physical SM ID, compares event-timed GEMV variance and checks CUPTI kernel intervals.

[Architecture](docs/ARCHITECTURE.md) · [Green Context API notes](docs/GREEN_CONTEXT_SPEC.md) · [Apache 2.0 license](LICENSE)
