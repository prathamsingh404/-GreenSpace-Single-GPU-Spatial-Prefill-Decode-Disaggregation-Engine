"""
Master EngineCore Orchestrator for Project GreenSpace.
Forks the vLLM V1 decoupled architecture into a high-performance,
dual-thread spatial prefill-decode disaggregation engine on a single GPU.
"""

import logging
import uuid
import time
from typing import List, Optional, Dict, Any
import torch

from greenspace.core.config import EngineConfig
from greenspace.core.request import Request, RequestStatus
from greenspace.core.block_manager import BlockManager
from greenspace.core.paged_kv_cache import PagedKVCache
from greenspace.models.transformer import GreenSpaceTransformerModel
from greenspace.scheduler.dual_scheduler import DualQueueScheduler
from greenspace.cuda.driver import driver
from greenspace.cuda.green_context import GreenContext
from greenspace.hardware.green_context_pool import GreenContextLattice
from greenspace.cuda.event_coordinator import EventCoordinator
from greenspace.core.controller import ClosedLoopPIDController
from greenspace.core.prefill_executor import PrefillExecutor
from greenspace.core.decode_executor import DecodeExecutor

logger = logging.getLogger("GreenSpace.EngineCore")


class GreenSpaceEngineCore:
    """
    Main entry point for Project GreenSpace.
    Coordinates hardware SM slicing, zero-copy HBM handoffs, dual concurrent
    submission loops, and closed-loop PID bandwidth throttling.
    """
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.device_id = self.config.greenspace.device_id

        # 1. Discover GPU hardware topology and SM count
        self.gpu_info = driver.get_device_info(self.device_id)
        self.total_sms = self.gpu_info["sm_count"]
        gs = self.config.greenspace
        self.alignment = self.gpu_info["sm_coscheduled_alignment"]
        self.lattice = None
        p_stream = d_stream = None
        if gs.enable_green_contexts:
            if not (0 < gs.sm_partition_prefill < 1 and 0 < gs.sm_partition_decode < 1
                    and abs(gs.sm_partition_prefill + gs.sm_partition_decode - 1) < 1e-6):
                raise ValueError("Prefill and decode SM shares must be positive and sum to one")
            self.lattice = GreenContextLattice(
                self.device_id, ratios=(.2, .3, .4, .5, .6, gs.sm_partition_decode),
                prefill_limit=gs.workqueue_concurrency_limit_prefill,
                decode_limit=gs.workqueue_concurrency_limit_decode,
                prefill_priority=gs.stream_priority_prefill, decode_priority=gs.stream_priority_decode)
            pair = self.lattice.get_pair(gs.sm_partition_decode)
            self.prefill_context, self.decode_context = pair.prefill, pair.decode
            p_stream, d_stream = pair.prefill_stream, pair.decode_stream
        else:
            self.prefill_context = GreenContext(self.total_sms, self.device_id,
                priority=gs.stream_priority_prefill, enable_hardware_isolation=False)
            self.decode_context = GreenContext(self.total_sms, self.device_id,
                priority=gs.stream_priority_decode, enable_hardware_isolation=False)
        self.prefill_sms = self.prefill_context.num_sms
        self.decode_sms = self.decode_context.num_sms

        # 3. Initialize Shared GPU HBM Paged KV Cache and Block Manager
        self.kv_cache = PagedKVCache(self.config.model, self.config.cache, device=f"cuda:{self.device_id}")
        self.block_manager = BlockManager(self.config.cache)

        # 4. Initialize Neural Model
        self.model = GreenSpaceTransformerModel(self.config.model, device=f"cuda:{self.device_id}")

        # 5. Initialize Decoupled Dual Scheduler and Coordination Layer
        self.scheduler = DualQueueScheduler(max_num_seqs=self.config.max_num_seqs)
        self.event_coordinator = EventCoordinator(device_id=self.device_id)

        # 6. Initialize Closed-Loop PID Controller
        self.controller = ClosedLoopPIDController(
            slo_config=self.config.slo,
            greenspace_config=self.config.greenspace,
            prefill_context=self.prefill_context,
            decode_context=self.decode_context,
            total_gpu_sms=self.total_sms
        )

        # 7. Initialize Dual-Thread Execution Engine
        self.prefill_executor = PrefillExecutor(
            config=self.config,
            model=self.model,
            kv_cache=self.kv_cache,
            block_manager=self.block_manager,
            scheduler=self.scheduler,
            prefill_context=self.prefill_context,
            event_coordinator=self.event_coordinator,
            controller=self.controller, stream=p_stream
        )
        self.decode_executor = DecodeExecutor(
            config=self.config,
            model=self.model,
            kv_cache=self.kv_cache,
            block_manager=self.block_manager,
            scheduler=self.scheduler,
            decode_context=self.decode_context,
            event_coordinator=self.event_coordinator,
            controller=self.controller, stream=d_stream
        )

        self._is_started = False
        self._closed = False
        init_stream = torch.cuda.current_stream(self.device_id)
        self.prefill_executor.stream.torch_stream.wait_stream(init_stream)
        self.decode_executor.stream.torch_stream.wait_stream(init_stream)
        self.warmup()

    @torch.inference_mode()
    def warmup(self):
        """Pre-warms cuBLAS, CUDA kernels, and memory pools to eliminate cold-start jitter."""
        logger.info("Warming up GreenSpace execution pipelines...")
        try:
            warmup_req = Request("warmup", [1, 2], max_tokens=2)
            b = self.block_manager.allocate_for_prompt(warmup_req)
            with torch.cuda.stream(self.prefill_executor.stream.torch_stream):
                _ = self.model.forward_prefill([1, 2], b, self.kv_cache)
            self.prefill_executor.stream.synchronize()

            with torch.cuda.stream(self.decode_executor.stream.torch_stream):
                _ = self.model.forward_decode(
                    token_id=2,
                    block_ids=b,
                    total_tokens=2,
                    target_block_id=b[0],
                    target_slot_idx=1,
                    kv_cache=self.kv_cache
                )
            self.decode_executor.stream.synchronize()
        finally:
            self.block_manager.free(warmup_req)

    def start(self):
        """Launches the dual concurrent CPU execution loops."""
        if self._closed:
            raise RuntimeError("Engine has been closed")
        if not self._is_started:
            self.prefill_executor.start()
            self.decode_executor.start()
            self._is_started = True
            logger.info("Project GreenSpace EngineCore running.")

    def stop(self):
        """Gracefully terminates executor threads and cleans up GPU resources."""
        if self._is_started:
            self.prefill_executor.stop()
            self.decode_executor.stop()
        for request in self.scheduler.drain():
            self.event_coordinator.release_request(request)
            self.block_manager.free(request)
            self.scheduler.mark_finished(request, RuntimeError("Engine stopped before completion"))
        if self.lattice is not None:
            self.lattice.close()
        else:
            self.prefill_context.close()
            self.decode_context.close()
        self._closed = True
        self._is_started = False

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 128,
        request_id: Optional[str] = None
    ) -> Request:
        """Submits a new inference request to the Prefill queue."""
        if self._closed:
            raise RuntimeError("Engine has been closed")
        if len(prompt_token_ids) + max_tokens - 1 > self.config.max_model_len:
            raise ValueError("Request exceeds max_model_len")
        if any(token < 0 or token >= self.config.model.vocab_size for token in prompt_token_ids):
            raise ValueError("Prompt token ID is outside the model vocabulary")
        req_id = request_id or str(uuid.uuid4())
        req = Request(request_id=req_id, prompt_token_ids=prompt_token_ids, max_tokens=max_tokens)
        self.scheduler.add_request(req)
        return req

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive system telemetry and latency statistics."""
        samples = list(self.decode_executor._recent_tpots)
        rolling_tpot = sum(samples) / len(samples) if samples else 0.0
        return {
            "gpu_name": self.gpu_info["name"],
            "total_sms": self.total_sms,
            "prefill_sms": self.prefill_sms,
            "decode_sms": self.decode_sms,
            "prefills_completed": self.prefill_executor.total_prefills_processed,
            "tokens_generated": self.decode_executor.total_tokens_generated,
            "current_rolling_tpot_ms": rolling_tpot,
            "target_tpot_ms": self.config.slo.target_tpot_ms,
            "hardware_isolation": self.prefill_context.green_ctx_handle is not None,
            "configured_prefill_workqueue_limit": self.prefill_context.configured_workqueue_limit,
            "current_workqueue_limit": self.controller.current_wq_limit,
            "requested_full_sm": self.controller.is_elastic_full_sm,
            "is_elastic_full_sm": self.lattice is None,
            "active_decodes": self.scheduler.num_active_decodes,
            "pending_prefills": self.scheduler.num_pending_prefills,
            "free_kv_blocks": self.block_manager.num_free_blocks,
            "total_decode_steps": self.decode_executor.step_count,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

