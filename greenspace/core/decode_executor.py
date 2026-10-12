"""
Decode Thread Executor for Project GreenSpace.
Runs a continuous, non-blocking autoregressive generation loop bound to the Decode Green Context stream,
fetching from shared HBM KV Cache, enforcing memory barrier waits, and feeding latency telemetry to the PID controller.
"""

import threading
import time
import logging
from collections import deque
from typing import Optional, List, Deque
import torch

from greenspace.core.config import EngineConfig
from greenspace.core.request import Request, RequestStatus
from greenspace.core.block_manager import BlockManager
from greenspace.core.paged_kv_cache import PagedKVCache
from greenspace.models.transformer import GreenSpaceTransformerModel
from greenspace.scheduler.dual_scheduler import DualQueueScheduler
from greenspace.cuda.green_context import GreenContext, GreenStream
from greenspace.cuda.event_coordinator import EventCoordinator
from greenspace.core.controller import ClosedLoopPIDController

logger = logging.getLogger("GreenSpace.Decode")


class DecodeExecutor:
    """
    Dedicated worker thread driving the Decode Green Context (memory-bound partition).
    Continuously executes autoregressive token generation across all active sequences,
    enforcing zero-copy CUDA event barriers and calculating per-token TPOT telemetry.
    """
    def __init__(
        self,
        config: EngineConfig,
        model: GreenSpaceTransformerModel,
        kv_cache: PagedKVCache,
        block_manager: BlockManager,
        scheduler: DualQueueScheduler,
        decode_context: GreenContext,
        event_coordinator: EventCoordinator,
        controller: Optional[ClosedLoopPIDController] = None
    ):
        self.config = config
        self.model = model
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.scheduler = scheduler
        self.decode_context = decode_context
        self.event_coordinator = event_coordinator
        self.controller = controller

        # Allocate dedicated Green Context stream for decode
        self.stream: GreenStream = self.decode_context.Stream()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.step_count = 0
        self.total_tokens_generated = 0

        # Rolling window for TPOT telemetry
        self._recent_tpots: Deque[float] = deque(maxlen=20)
        self.step_latencies: List[float] = []

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="Decode-Thread", daemon=True)
        self._thread.start()
        logger.info(f"Decode Executor started on stream {self.stream}")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Decode Executor stopped.")

    def _run_loop(self):
        torch.cuda.set_device(self.config.greenspace.device_id)
        while self._running:
            active_batch = self.scheduler.get_active_decode_batch()
            if not active_batch:
                # Inform controller of idle decode state for elastic SM rebalancing
                if self.controller and self.step_count % 10 == 0:
                    self.controller.update_telemetry(0.0, 0, self.step_count)
                time.sleep(0.005)
                continue

            t_step_start = time.perf_counter()
            self._step(active_batch)
            t_step_end = time.perf_counter()

            step_duration_ms = (t_step_end - t_step_start) * 1000.0
            self.step_latencies.append(step_duration_ms)
            self._recent_tpots.append(step_duration_ms)

            self.step_count += 1

            # Closed-Loop PID Telemetry feedback every N steps
            if self.controller and (self.step_count % self.config.slo.telemetry_window_steps == 0):
                rolling_tpot = sum(self._recent_tpots) / len(self._recent_tpots) if self._recent_tpots else 0.0
                self.controller.update_telemetry(
                    rolling_tpot_ms=rolling_tpot,
                    num_active_decodes=len(active_batch),
                    step=self.step_count
                )

    def _step(self, batch: List[Request]):
        """Executes one autoregressive generation step for all requests in the active batch."""
        for request in batch:
            # Step 1: Memory Barrier Check for requests arriving from Prefill
            if request.status == RequestStatus.READ_ONLY_READY:
                # Non-blocking GPU-side wait on prefill completion event
                self.event_coordinator.acquire_for_decode(request, self.stream)

            # Step 2: Allocate slot in PagedKVCache for the next token
            target_block_id, target_slot_idx = self.block_manager.append_slot_for_decode(request)

            # Step 3: Single-token forward pass on Decode Green Context
            last_token = request.generated_token_ids[-1]
            total_tokens = request.total_tokens

            with torch.cuda.stream(self.stream.torch_stream):
                logits = self.model.forward_decode(
                    token_id=last_token,
                    block_ids=request.allocated_blocks,
                    total_tokens=total_tokens,
                    target_block_id=target_block_id,
                    target_slot_idx=target_slot_idx,
                    kv_cache=self.kv_cache
                )
                # Greedy/argmax sampling
                next_token = int(torch.argmax(logits, dim=-1).item())

            # Synchronize host to measure precise per-token completion time
            self.stream.synchronize()

            request.record_generated_token(next_token)
            self.total_tokens_generated += 1

            # Step 4: Completion check
            if request.is_finished():
                request.finish_time = time.perf_counter()
                self.event_coordinator.release_request(request)
                self.block_manager.free(request)
                self.scheduler.mark_finished(request)

