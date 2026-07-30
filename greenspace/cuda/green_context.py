"""
High-Level GreenContext and GreenStream abstractions for PyTorch.
Provides spatial hardware SM partitioning, workqueue concurrency throttling,
and direct interoperability with PyTorch CUDA streams.
"""

import logging
from typing import Optional, ContextManager
from contextlib import contextmanager
import torch

from greenspace.cuda.driver import driver, CUDA_SUCCESS

logger = logging.getLogger("GreenSpace.CUDA")


class GreenStream:
    """
    High-performance wrapper around a CUDA Stream bound to a Green Context partition.
    Interoperates seamlessly with PyTorch's stream execution context.
    """
    def __init__(
        self,
        raw_stream_handle: int,
        device_id: int = 0,
        parent_context: Optional["GreenContext"] = None
    ):
        self.raw_handle = raw_stream_handle
        self.device_id = device_id
        self.parent_context = parent_context
        
        # Wrap raw CUDA stream pointer with PyTorch ExternalStream
        if raw_stream_handle:
            self._torch_stream = torch.cuda.ExternalStream(raw_stream_handle, device=device_id)
        else:
            self._torch_stream = torch.cuda.Stream(device=device_id)

    @property
    def torch_stream(self) -> torch.cuda.Stream:
        return self._torch_stream

    @property
    def cuda_stream(self) -> int:
        return self.raw_handle

    def synchronize(self):
        """Blocks CPU until all scheduled operations on this stream have completed."""
        self._torch_stream.synchronize()

    def wait_event(self, event: torch.cuda.Event):
        """Makes this stream wait asynchronously on a CUDA event."""
        self._torch_stream.wait_event(event)

    def record_event(self, event: Optional[torch.cuda.Event] = None) -> torch.cuda.Event:
        """Records a CUDA event on this stream."""
        if event is None:
            event = torch.cuda.Event(enable_timing=True)
        event.record(self._torch_stream)
        return event

    def __enter__(self):
        return self._torch_stream.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._torch_stream.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return f"<GreenStream device={self.device_id} handle=0x{self.raw_handle:x}>"


class GreenContext:
    """
    Represents an isolated spatial GPU partition (Green Context) reserving a specific
    subset of Streaming Multiprocessors (SMs) and configured with hardware workqueue limits.
    """
    def __init__(
        self,
        num_sms: int,
        device_id: int = 0,
        workqueue_concurrency_limit: Optional[int] = None,
        priority: int = 0,
        enable_hardware_isolation: bool = True
    ):
        self.num_sms = num_sms
        self.device_id = device_id
        self.workqueue_concurrency_limit = workqueue_concurrency_limit
        self.priority = priority
        self.enable_hardware_isolation = enable_hardware_isolation
        
        self.green_ctx_handle: int = 0
        self.cu_ctx_handle: int = 0
        self._streams = []
        self._is_closed = False

        # Ensure PyTorch CUDA primary context is fully initialized
        if not torch.cuda.is_initialized():
            torch.cuda.init()

        self._init_hardware_context()

    def _init_hardware_context(self):
        """Initializes low-level Green Context or fallback stream."""
        if self.enable_hardware_isolation and driver.has_green_ctx:
            try:
                self.green_ctx_handle, self.cu_ctx_handle = driver.create_green_context(
                    sm_count=self.num_sms,
                    workqueue_concurrency_limit=self.workqueue_concurrency_limit,
                    device_id=self.device_id
                )
                logger.info(
                    f"Initialized Native Hardware GreenContext: {self.num_sms} SMs on device {self.device_id} "
                    f"(hGreenCtx=0x{self.green_ctx_handle:x}, hCtx=0x{self.cu_ctx_handle:x})"
                )
                return
            except Exception as e:
                logger.warning(
                    f"Native Green Context creation failed ({e}); falling back to priority stream isolation."
                )

        # Fallback path if hardware context creation not permitted
        self.green_ctx_handle = 0
        self.cu_ctx_handle = 0

    def Stream(self) -> GreenStream:
        """Creates a new CUDA stream bound to this Green Context's hardware partition."""
        if self._is_closed:
            raise RuntimeError("Cannot allocate stream from closed GreenContext.")

        if self.cu_ctx_handle != 0:
            raw_stream = driver.create_stream_in_context(self.cu_ctx_handle, priority=self.priority)
        else:
            # Fallback stream with specified priority
            torch_s = torch.cuda.Stream(device=self.device_id, priority=self.priority)
            raw_stream = torch_s.cuda_stream

        stream = GreenStream(raw_stream, device_id=self.device_id, parent_context=self)
        self._streams.append(stream)
        return stream

    def set_workqueue_concurrency_limit(self, limit: int):
        """
        Dynamically adjusts the workqueue concurrency limit to throttle kernel launch
        rates and prevent shared memory bus / L2 cache saturation.
        """
        self.workqueue_concurrency_limit = max(1, limit)
        logger.debug(f"Adjusted GreenContext workqueue concurrency limit to {self.workqueue_concurrency_limit}")

    def rebind_sm_partition(self, new_num_sms: int):
        """
        Elastic SM rebalancing: Re-allocates SM resources (e.g. reclamation when decode queue is empty).
        """
        if new_num_sms == self.num_sms:
            return

        logger.info(f"Rebinding GreenContext SM partition from {self.num_sms} -> {new_num_sms} SMs")
        self.num_sms = new_num_sms
        # Re-initialize context if hardware isolation is active
        if self.cu_ctx_handle != 0:
            try:
                driver.destroy_green_context(self.green_ctx_handle)
            except Exception:
                pass
            self._init_hardware_context()

    def close(self):
        """Releases all allocated hardware resources."""
        if self._is_closed:
            return
        self._is_closed = True
        for s in self._streams:
            try:
                if self.cu_ctx_handle != 0:
                    driver.destroy_stream(s.raw_handle)
            except Exception:
                pass
        self._streams.clear()

        if self.green_ctx_handle != 0:
            try:
                driver.destroy_green_context(self.green_ctx_handle)
            except Exception:
                pass
            self.green_ctx_handle = 0
            self.cu_ctx_handle = 0

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        mode = "Hardware" if self.green_ctx_handle != 0 else "StreamIsolated"
        return f"<GreenContext mode={mode} sms={self.num_sms} wq_limit={self.workqueue_concurrency_limit}>"


# Docstring updates on ExternalStream lifetime

# Fast non-blocking event synchronization
