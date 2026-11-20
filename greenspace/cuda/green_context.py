"""PyTorch stream views over explicitly owned CUDA Green Context streams."""

from typing import Optional

import torch
from cuda.bindings import driver as cu

from greenspace.cuda.driver import checked, driver


class GreenStream:
    def __init__(self, raw_stream_handle: int, device_id: int = 0,
                 parent_context: Optional["GreenContext"] = None):
        self.raw_handle = raw_stream_handle
        self.device_id = device_id
        self.parent_context = parent_context
        self._torch_stream = torch.cuda.ExternalStream(raw_stream_handle, device=device_id)

    @property
    def torch_stream(self) -> torch.cuda.Stream:
        if self.parent_context is not None and self.parent_context.closed:
            raise RuntimeError("The stream's Green Context is closed")
        return self._torch_stream

    @property
    def cuda_stream(self) -> int:
        return self.torch_stream.cuda_stream

    def synchronize(self):
        self.torch_stream.synchronize()

    def wait_event(self, event: torch.cuda.Event):
        self.torch_stream.wait_event(event)

    def record_event(self, event: Optional[torch.cuda.Event] = None) -> torch.cuda.Event:
        return self.torch_stream.record_event(event)

    def __enter__(self):
        self._guard = torch.cuda.stream(self.torch_stream)
        return self._guard.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._guard.__exit__(exc_type, exc_val, exc_tb)


class GreenContext:
    """Own one driver resource. Paired users must use the same split operation."""

    def __init__(self, num_sms: int, device_id: int = 0,
                 workqueue_concurrency_limit: int | None = None, priority: int = 0,
                 enable_hardware_isolation: bool = True, *, resource=None):
        self.closed = False
        self.green_ctx_handle = None
        self._streams = []
        self.device_id = device_id
        self.priority = priority
        self.workqueue_concurrency_limit = workqueue_concurrency_limit
        self.configured_workqueue_limit = None
        with torch.cuda.device(device_id):
            torch.cuda.init()
            if enable_hardware_isolation:
                if resource is None:
                    resource = driver.sm_resource(device_id)
                    if num_sms != resource.sm.smCount:
                        _, resource = driver.split(device_id, num_sms)
                self.num_sms = resource.sm.smCount
                self.green_ctx_handle, self.configured_workqueue_limit = driver.create_green_context(
                    resource, device_id, workqueue_concurrency_limit)
            else:
                self.num_sms = torch.cuda.get_device_properties(device_id).multi_processor_count

    def Stream(self) -> GreenStream:
        if self.closed:
            raise RuntimeError("Cannot allocate a stream from a closed context")
        if self.green_ctx_handle is None:
            stream = torch.cuda.Stream(device=self.device_id, priority=self.priority)
            self._streams.append(stream)
            return GreenStream(stream.cuda_stream, self.device_id, self)
        handle = checked(cu.cuGreenCtxStreamCreate, self.green_ctx_handle,
                         int(cu.CUstream_flags.CU_STREAM_NON_BLOCKING), self.priority)
        self._streams.append(handle)
        return GreenStream(int(handle), self.device_id, self)

    def set_workqueue_concurrency_limit(self, limit: int):
        """Record a controller request. Hardware configuration is fixed at creation."""
        self.workqueue_concurrency_limit = max(1, limit)

    def rebind_sm_partition(self, new_num_sms: int):
        if new_num_sms != self.num_sms:
            raise RuntimeError("Select a pre-provisioned context pair at a synchronized phase boundary")

    def close(self):
        if self.closed:
            return
        while self._streams:
            stream = self._streams[-1]
            if self.green_ctx_handle is not None:
                checked(cu.cuStreamSynchronize, stream)
                checked(cu.cuStreamDestroy, stream)
            else:
                stream.synchronize()
            self._streams.pop()
        if self.green_ctx_handle is not None:
            checked(cu.cuGreenCtxDestroy, self.green_ctx_handle)
            self.green_ctx_handle = None
        self.closed = True

    def __del__(self):
        if getattr(self, "green_ctx_handle", None) is not None:
            self.close()
