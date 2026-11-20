"""
Decoupled Dual-Queue Scheduler for Spatial Prefill-Decode Disaggregation.
Maintains isolated queues for prefill prompt ingestion and continuous decode generation.
"""

import threading
from collections import deque
from typing import List, Optional, Deque
from greenspace.core.request import Request, RequestStatus


class DualQueueScheduler:
    """
    Decouples request scheduling into two independent execution streams:
    1. Prefill Queue: Waiting prompts awaiting forward-pass processing on Prefill Green Context.
    2. Decode Set: Active sequences generating tokens on Decode Green Context.
    """
    def __init__(self, max_num_seqs: int = 64):
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        self.max_num_seqs = max_num_seqs
        self._lock = threading.Lock()
        self._closed = False
        
        self._prefill_queue: Deque[Request] = deque()
        self._decode_pool: List[Request] = []
        self._live: dict[str, Request] = {}
        self._ready: Deque[Request] = deque()

    def add_request(self, request: Request):
        with self._lock:
            if self._closed:
                raise RuntimeError("Scheduler has been closed")
            if request.request_id in self._live:
                raise ValueError(f"Duplicate active request ID: {request.request_id}")
            self._live[request.request_id] = request
            request.status = RequestStatus.WAITING
            self._prefill_queue.append(request)

    def get_next_prefill_request(self) -> Optional[Request]:
        """Pulls next waiting prompt for Prefill Thread."""
        with self._lock:
            if self._prefill_queue and len(self._decode_pool) < self.max_num_seqs:
                req = self._prefill_queue.popleft()
                req.status = RequestStatus.PREFILLING
                return req
            return None

    def transfer_to_decode(self, request: Request):
        """Zero-copy transition: Enters request into active decode pool."""
        with self._lock:
            if len(self._decode_pool) < self.max_num_seqs:
                self._decode_pool.append(request)
            else:
                # Completed prompts retain their existing pages until decode admission.
                self._ready.append(request)

    def get_active_decode_batch(self) -> List[Request]:
        """Pulls all active decoding requests for the next autoregressive step."""
        with self._lock:
            return list(self._decode_pool)

    def mark_finished(self, request: Request, error: Exception | None = None):
        with self._lock:
            if request in self._decode_pool:
                self._decode_pool.remove(request)
                if self._ready:
                    self._decode_pool.append(self._ready.popleft())
            self._live.pop(request.request_id, None)
            request.error = str(error) if error is not None else None
            request.status = RequestStatus.FAILED if error is not None else RequestStatus.FINISHED

    def drain(self) -> List[Request]:
        """After both workers join, return requests whose resources need cleanup."""
        with self._lock:
            self._closed = True
            pending = list(self._live.values())
            self._live.clear()
            self._prefill_queue.clear()
            self._decode_pool.clear()
            self._ready.clear()
            return pending

    @property
    def has_pending_prefills(self) -> bool:
        with self._lock:
            return len(self._prefill_queue) > 0

    @property
    def num_active_decodes(self) -> int:
        with self._lock:
            return len(self._decode_pool)

    @property
    def num_pending_prefills(self) -> int:
        with self._lock:
            return len(self._prefill_queue)

