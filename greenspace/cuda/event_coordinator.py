"""
CUDA Event Coordinator for Zero-Copy HBM Handoff between Prefill and Decode Green Contexts.
Guarantees memory coherence and non-blocking GPU-side barrier synchronization.
"""

import threading
import time
from typing import Optional, Dict
import torch

from greenspace.core.request import Request, RequestStatus
from greenspace.cuda.green_context import GreenStream


class EventCoordinator:
    """
    Coordinates GPU-side memory barriers and state transitions between
    Prefill and Decode streams.
    """
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._lock = threading.RLock()
        self._event_pool = []
        self._active_events: Dict[str, torch.cuda.Event] = {}

    def _get_event(self) -> torch.cuda.Event:
        with self._lock:
            if self._event_pool:
                return self._event_pool.pop()
            return torch.cuda.Event(enable_timing=True, blocking=False)

    def _recycle_event(self, event: torch.cuda.Event):
        with self._lock:
            if len(self._event_pool) < 256:
                self._event_pool.append(event)

    def signal_prefill_completed(self, request: Request, prefill_stream: GreenStream) -> torch.cuda.Event:
        """
        Called by Prefill Thread when all prompt forward passes and KV writes have completed.
        Records a CUDA event on the Prefill stream and marks the request as READ_ONLY_READY.
        """
        event = self._get_event()
        # Record event on the prefill stream ensuring all memory stores to KV cache pages are committed
        event.record(prefill_stream.torch_stream)
        
        with self._lock:
            self._active_events[request.request_id] = event
            request.prefill_event = event
            request.status = RequestStatus.READ_ONLY_READY
            request.prefill_end_time = time.perf_counter()

        return event

    def acquire_for_decode(self, request: Request, decode_stream: GreenStream):
        """
        Called by Decode Thread before reading prompt KV cache pages for the first decode step.
        Instructs the Decode stream to wait for the prefill completion event on the GPU.
        This operation is completely non-blocking to the CPU!
        """
        with self._lock:
            event = self._active_events.get(request.request_id)
            if event is None:
                raise RuntimeError(f"No prefill completion event registered for request {request.request_id}")

            # Issue stream wait: Decode stream will not dispatch subsequent kernels until event signals
            decode_stream.wait_event(event)
            request.status = RequestStatus.DECODING
            request.decode_start_time = time.perf_counter()

    def release_request(self, request: Request):
        """Releases the CUDA event associated with the completed request."""
        with self._lock:
            event = self._active_events.pop(request.request_id, None)
            if event is not None:
                self._recycle_event(event)
                request.prefill_event = None

    def get_handoff_latency_ms(self, request: Request) -> float:
        """
        Calculates host-side handoff latency in milliseconds between prefill completion
        and decode acquisition.
        """
        if request.prefill_end_time and request.decode_start_time:
            return max(0.0, (request.decode_start_time - request.prefill_end_time) * 1000.0)
        return 0.0


# Atomic state transition helpers

# Use re-entrant lock to prevent self-deadlock
