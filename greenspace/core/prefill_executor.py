"""
Prefill Thread Executor for Project GreenSpace.
Runs a dedicated CPU submission loop bound to the Prefill Green Context stream,
processing prompts, populating Paged KV Cache blocks, and signaling memory barrier events.
"""

import threading
import time
import logging
from typing import Optional
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

logger = logging.getLogger("GreenSpace.Prefill")


class PrefillExecutor:
    """
    Dedicated worker thread driving the Prefill Green Context (compute-bound partition).
    Ingests prompts, executes dense forward passes, writes KV cache into GPU HBM,
    and signals zero-copy handoff events.
    """
    def __init__(
        self,
        config: EngineConfig,
        model: GreenSpaceTransformerModel,
        kv_cache: PagedKVCache,
        block_manager: BlockManager,
        scheduler: DualQueueScheduler,
        prefill_context: GreenContext,
        event_coordinator: EventCoordinator,
        controller: Optional[ClosedLoopPIDController] = None
    ):
        self.config = config
        self.model = model
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.scheduler = scheduler
        self.prefill_context = prefill_context
        self.event_coordinator = event_coordinator
        self.controller = controller

        # Allocate dedicated Green Context stream for prefill
        self.stream: GreenStream = self.prefill_context.Stream()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.total_prefills_processed = 0
        self.total_prefill_tokens_processed = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="Prefill-Thread", daemon=True)
        self._thread.start()
        logger.info(f"Prefill Executor started on stream {self.stream}")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Prefill Executor stopped.")

    def _run_loop(self):
        torch.cuda.set_device(self.config.greenspace.device_id)
        while self._running:
            request = self.scheduler.get_next_prefill_request()
            if request is None:
                # Idle backoff
                time.sleep(0.005)
                continue

            try:
                self._process_prefill(request)
            except Exception as e:
                logger.error(f"Error processing prefill for request {request.request_id}: {e}", exc_info=True)
                self.block_manager.free(request)
                self.scheduler.mark_finished(request)

    def _process_prefill(self, request: Request):
        request.prefill_start_time = time.perf_counter()

        # Step 1: Allocate physical KV cache blocks in GPU HBM
        block_ids = self.block_manager.allocate_for_prompt(request)

        # Step 2: Forward pass on dedicated Prefill Green Context stream
        with torch.cuda.stream(self.stream.torch_stream):
            logits = self.model.forward_prefill(
                prompt_token_ids=request.prompt_token_ids,
                block_ids=block_ids,
                kv_cache=self.kv_cache
            )
            # Sample first token
            first_token = int(torch.argmax(logits, dim=-1).item())

        # Step 3: Zero-Copy Synchronization Barrier
        # Record cudaEvent on the prefill stream ensuring all KV stores are retired to HBM
        self.event_coordinator.signal_prefill_completed(request, self.stream)

        # Record generated token
        request.record_generated_token(first_token)

        # Step 4: Instantly pass pointer to Decode queue (zero data copied!)
        self.scheduler.transfer_to_decode(request)

        self.total_prefills_processed += 1
        self.total_prefill_tokens_processed += request.num_prompt_tokens

