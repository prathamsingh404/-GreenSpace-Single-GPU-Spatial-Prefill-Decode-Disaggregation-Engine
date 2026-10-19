"""
Profiler integration for NVIDIA Nsight Systems (nsys), Nsight Compute (ncu),
and PyTorch Profiler traces with NVTX range annotations.
"""

import os
from typing import Optional, ContextManager
from contextlib import contextmanager
import torch


class NsightProfiler:
    """
    Manages NVTX instrumentation and execution trace capture for
    NVIDIA Nsight Systems and PyTorch Profiler.
    """
    @staticmethod
    def push_range(name: str, color_argb: Optional[int] = None):
        """Pushes an NVTX marker for NVIDIA Nsight timeline visualization."""
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)

    @staticmethod
    def pop_range():
        """Pops the current NVTX range marker."""
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

    @staticmethod
    @contextmanager
    def profile_scope(name: str):
        NsightProfiler.push_range(name)
        try:
            yield
        finally:
            NsightProfiler.pop_range()

    @staticmethod
    def start_trace(output_dir: str = "traces") -> torch.profiler.profile:
        """Starts a full PyTorch CUDA execution trace capturing SM activity."""
        os.makedirs(output_dir, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        prof.start()
        return prof

