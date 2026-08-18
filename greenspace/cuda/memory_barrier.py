"""
GPU Memory Barrier and Memory Coherence Primitives for Zero-Copy HBM Disaggregation.
"""

import torch
from typing import Tuple


class GPUMemoryBarrier:
    """
    Ensures memory store-load ordering between prefill and decode execution zones.
    """
    @staticmethod
    def insert_write_barrier(stream: torch.cuda.Stream) -> torch.cuda.Event:
        """
        Inserts a GPU memory write barrier on the source stream by recording a completion event.
        Guarantees that all preceding HBM writes (e.g., KV cache blocks) are globally retired.
        """
        event = torch.cuda.Event(enable_timing=True, blocking=False)
        event.record(stream)
        return event

    @staticmethod
    def wait_read_barrier(stream: torch.cuda.Stream, event: torch.cuda.Event):
        """
        Inserts a read barrier on the target stream. The GPU will not execute subsequent
        kernels on this stream until the memory barrier event has completed on the source stream.
        """
        stream.wait_event(event)

    @staticmethod
    def verify_coherence(
        producer_stream: torch.cuda.Stream,
        consumer_stream: torch.cuda.Stream,
        tensor_size: int = 1048576  # 1M elements
    ) -> Tuple[bool, float]:
        """
        Microbenchmark test that writes a test pattern in producer_stream,
        signals a memory barrier event, waits on consumer_stream, and verifies
        bit-exact coherence with zero CPU synchronization in the middle.
        Returns (is_coherent, barrier_latency_us).
        """
        # Allocate shared HBM buffer
        buffer = torch.zeros(tensor_size, dtype=torch.int32, device='cuda')
        pattern = torch.randint(1, 1000000, (tensor_size,), dtype=torch.int32, device='cuda')
        result = torch.zeros(tensor_size, dtype=torch.int32, device='cuda')

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(producer_stream):
            start_event.record(producer_stream)
            # Write pattern into shared buffer
            buffer.copy_(pattern)
            barrier_event = GPUMemoryBarrier.insert_write_barrier(producer_stream)

        with torch.cuda.stream(consumer_stream):
            # Wait on GPU barrier without CPU intervention
            GPUMemoryBarrier.wait_read_barrier(consumer_stream, barrier_event)
            # Read from shared buffer
            result.copy_(buffer)
            end_event.record(consumer_stream)

        # Synchronize host to read verification status
        producer_stream.synchronize()
        consumer_stream.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        is_coherent = bool(torch.equal(pattern, result))

        return is_coherent, elapsed_ms * 1000.0  # microseconds

