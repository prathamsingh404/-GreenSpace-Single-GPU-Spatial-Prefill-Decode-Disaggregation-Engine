"""
Baseline Monolithic Engine (Simulating standard vLLM execution).
Executes prefill prompts and decode steps sequentially on a single monolithic CUDA stream
without spatial SM partitioning, illustrating prefill-decode interference and TPOT spikes.
"""

import time
import logging
import uuid
from typing import List, Optional, Dict, Any
from collections import deque
import torch

from greenspace.core.config import EngineConfig
from greenspace.core.request import Request, RequestStatus
from greenspace.core.block_manager import BlockManager
from greenspace.core.paged_kv_cache import PagedKVCache
from greenspace.models.transformer import GreenSpaceTransformerModel

logger = logging.getLogger("Baseline.MonolithicEngine")


class MonolithicEngine:
    """
    Standard single-stream monolithic inference engine.
    Demonstrates the fundamental architectural bottleneck: compute-heavy prefill bursts
    hog SMs and stall active decode streams, causing extreme TPOT spikes.
    """
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.device = f"cuda:{self.config.greenspace.device_id}"

        self.kv_cache = PagedKVCache(self.config.model, self.config.cache, device=self.device)
        self.block_manager = BlockManager(self.config.cache)
        self.model = GreenSpaceTransformerModel(self.config.model, device=self.device)

        # Monolithic default stream (no green context isolation)
        self.stream = torch.cuda.default_stream()

        self.waiting_queue: deque[Request] = deque()
        self.active_decode_pool: List[Request] = []
        self.finished_requests: List[Request] = []

        self.step_latencies: List[float] = []
        self.step_count = 0

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 128,
        request_id: Optional[str] = None
    ) -> Request:
        req_id = request_id or str(uuid.uuid4())[:8]
        req = Request(request_id=req_id, prompt_token_ids=prompt_token_ids, max_tokens=max_tokens)
        self.waiting_queue.append(req)
        return req

    def step(self):
        """
        Single monolithic execution step.
        If a new prompt is in the waiting queue, it processes prefill FIRST,
        blocking all active decodes and inducing severe TPOT latency spikes!
        """
        t_start = time.perf_counter()

        # Prioritize prefill when arriving (monolithic behavior)
        if self.waiting_queue:
            req = self.waiting_queue.popleft()
            req.status = RequestStatus.PREFILLING
            req.prefill_start_time = time.perf_counter()

            # Allocate blocks
            block_ids = self.block_manager.allocate_for_prompt(req)

            # Heavy compute-bound prefill on monolithic stream
            with torch.cuda.stream(self.stream):
                logits = self.model.forward_prefill(
                    prompt_token_ids=req.prompt_token_ids,
                    block_ids=block_ids,
                    kv_cache=self.kv_cache
                )
                first_token = int(torch.argmax(logits, dim=-1).item())

            self.stream.synchronize()
            req.prefill_end_time = time.perf_counter()
            req.record_generated_token(first_token)
            req.status = RequestStatus.DECODING
            self.active_decode_pool.append(req)

        # Process active decodes
        if self.active_decode_pool:
            for req in list(self.active_decode_pool):
                target_block, target_slot = self.block_manager.append_slot_for_decode(req)
                last_token = req.generated_token_ids[-1]

                with torch.cuda.stream(self.stream):
                    logits = self.model.forward_decode(
                        token_id=last_token,
                        block_ids=req.allocated_blocks,
                        total_tokens=req.total_tokens,
                        target_block_id=target_block,
                        target_slot_idx=target_slot,
                        kv_cache=self.kv_cache
                    )
                    next_token = int(torch.argmax(logits, dim=-1).item())

                self.stream.synchronize()
                req.record_generated_token(next_token)

                if req.is_finished():
                    req.finish_time = time.perf_counter()
                    self.block_manager.free(req)
                    self.active_decode_pool.remove(req)
                    self.finished_requests.append(req)

        t_end = time.perf_counter()
        step_ms = (t_end - t_start) * 1000.0
        self.step_latencies.append(step_ms)
        self.step_count += 1

