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
        self.max_num_seqs = max_num_seqs
        self._lock = threading.Lock()
        
        self._prefill_queue: Deque[Request] = deque()
        self._decode_pool: List[Request] = []
        self._finished_requests: List[Request] = []

    def add_request(self, request: Request):
        with self._lock:
            request.status = RequestStatus.WAITING
            self._prefill_queue.append(request)

    def get_next_prefill_request(self) -> Optional[Request]:
        """Pulls next waiting prompt for Prefill Thread."""
        with self._lock:
            if self._prefill_queue:
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
                # If decode pool full, requeue at head
                self._prefill_queue.appendleft(request)

    def get_active_decode_batch(self) -> List[Request]:
        """Pulls all active decoding requests for the next autoregressive step."""
        with self._lock:
            return list(self._decode_pool)

    def mark_finished(self, request: Request):
        with self._lock:
            if request in self._decode_pool:
                self._decode_pool.remove(request)
            request.status = RequestStatus.FINISHED
            self._finished_requests.append(request)

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


# Atomic pointer handoff
