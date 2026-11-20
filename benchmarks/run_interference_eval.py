"""
Memory Bandwidth Contention and L2 Cache Thrashing Evaluation.
Quantifies the isolation efficacy of Green Contexts and workqueue concurrency limits
under increasing prefill matrix multiplication intensity.
"""

import os
import json
import time
import logging
import torch

from greenspace.core.config import EngineConfig, GreenSpaceConfig, SLOConfig
from greenspace.hardware.green_context_pool import GreenContextLattice
from greenspace.cuda.memory_barrier import GPUMemoryBarrier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GreenSpace.InterferenceEval")


def evaluate_memory_interference(device_id: int = 0):
    logger.info("Evaluating Memory Bandwidth Contention across SM Partitions...")
    torch.cuda.set_device(device_id)

    # Both configurations use complementary groups from a single split.
    initial = GreenContextLattice(device_id, ratios=(.4,), prefill_limit=8)
    throttled = GreenContextLattice(device_id, ratios=(.4,), prefill_limit=2)
    pair = initial.get_pair(.4)
    reduced_pair = throttled.get_pair(.4)
    if pair.prefill.configured_workqueue_limit == reduced_pair.prefill.configured_workqueue_limit:
        initial.close()
        throttled.close()
        raise RuntimeError("Driver configured the same limit for both conditions")
    s_prefill, s_decode = pair.prefill_stream, pair.decode_stream

    # Shared memory buffers
    buffer_size = 8 * 1024 * 1024  # 8M floats = 32 MB (matches GPU L2 cache size!)
    decode_buffer = torch.randn(buffer_size, dtype=torch.float16, device="cuda")
    prefill_matrix = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")

    torch.cuda.current_stream().synchronize()

    # 1. Baseline Decode Latency (No Prefill running)
    logger.info("Measuring isolated decode memory read latency (zero prefill contention)...")
    latencies_isolated = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.cuda.stream(s_decode.torch_stream):
            # Memory-bound vector reduction simulating KV cache scan
            _ = torch.sum(decode_buffer)
        s_decode.synchronize()
        latencies_isolated.append((time.perf_counter() - t0) * 1000.0)

    mean_isolated = sum(latencies_isolated[10:]) / len(latencies_isolated[10:])

    # 2. Decode Latency under Concurrent Unthrottled Prefill
    logger.info("Measuring decode latency under heavy unthrottled prefill GEMMs...")
    latencies_contended = []
    for _ in range(50):
        # Launch heavy prefill GEMMs on prefill stream
        with torch.cuda.stream(s_prefill.torch_stream):
            for _ in range(4):
                _ = torch.matmul(prefill_matrix, prefill_matrix)

        t0 = time.perf_counter()
        with torch.cuda.stream(s_decode.torch_stream):
            _ = torch.sum(decode_buffer)
        s_decode.synchronize()
        latencies_contended.append((time.perf_counter() - t0) * 1000.0)

    s_prefill.synchronize()
    mean_contended = sum(latencies_contended[10:]) / len(latencies_contended[10:])

    # 3. Decode Latency with GreenSpace Workqueue Throttling (Limit = 2)
    logger.info("Measuring decode latency with GreenSpace Workqueue Throttling active...")
    s_prefill, s_decode = reduced_pair.prefill_stream, reduced_pair.decode_stream
    latencies_throttled = []
    for _ in range(50):
        with torch.cuda.stream(s_prefill.torch_stream):
            for _ in range(4):
                _ = torch.matmul(prefill_matrix, prefill_matrix)

        t0 = time.perf_counter()
        with torch.cuda.stream(s_decode.torch_stream):
            _ = torch.sum(decode_buffer)
        s_decode.synchronize()
        latencies_throttled.append((time.perf_counter() - t0) * 1000.0)

    s_prefill.synchronize()
    mean_throttled = sum(latencies_throttled[10:]) / len(latencies_throttled[10:])

    logger.info("=" * 65)
    logger.info("MEMORY BANDWIDTH INTERFERENCE EVALUATION SUMMARY:")
    logger.info(f"Isolated Decode Latency:              {mean_isolated:.3f} ms (1.00x baseline)")
    logger.info(f"Unthrottled Concurrent Prefill:       {mean_contended:.3f} ms (Contention overhead: +{(mean_contended/mean_isolated - 1)*100:.1f}%)")
    logger.info(f"GreenSpace Throttled Prefill:         {mean_throttled:.3f} ms (Contention overhead: +{(mean_throttled/mean_isolated - 1)*100:.1f}%)")
    logger.info("=" * 65)

    initial.close()
    throttled.close()


if __name__ == "__main__":
    evaluate_memory_interference()

