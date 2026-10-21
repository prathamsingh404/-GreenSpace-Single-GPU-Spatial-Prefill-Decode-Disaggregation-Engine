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
from greenspace.cuda.green_context import GreenContext
from greenspace.cuda.memory_barrier import GPUMemoryBarrier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GreenSpace.InterferenceEval")


def evaluate_memory_interference(device_id: int = 0):
    logger.info("Evaluating Memory Bandwidth Contention across SM Partitions...")
    torch.cuda.set_device(device_id)

    # Allocate Prefill Green Context (12 SMs) and Decode Green Context (8 SMs)
    prefill_ctx = GreenContext(num_sms=12, device_id=device_id, workqueue_concurrency_limit=8)
    decode_ctx = GreenContext(num_sms=8, device_id=device_id, priority=-1)

    s_prefill = prefill_ctx.Stream()
    s_decode = decode_ctx.Stream()

    # Shared memory buffers
    buffer_size = 8 * 1024 * 1024  # 8M floats = 32 MB (matches GPU L2 cache size!)
    decode_buffer = torch.randn(buffer_size, dtype=torch.float16, device="cuda")
    prefill_matrix = torch.randn(2048, 2048, dtype=torch.float16, device="cuda")

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
    prefill_ctx.set_workqueue_concurrency_limit(2)
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

    prefill_ctx.close()
    decode_ctx.close()


if __name__ == "__main__":
    evaluate_memory_interference()

