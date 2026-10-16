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
        self.alignment = self.gpu_info["sm_coscheduled_alignment"]
        
        self.prefill_sms, self.decode_sms = self.config.greenspace.get_sm_counts(
            total_sms=self.total_sms,
            alignment=self.alignment
        )
        logger.info(
            f"Initializing Project GreenSpace on {self.gpu_info['name']} "
            f"({self.total_sms} SMs total). Slicing: {self.prefill_sms} SMs (Prefill Zone) / "
            f"{self.decode_sms} SMs (Decode Zone)."
        )

        # 2. Initialize Hardware Green Contexts
        self.prefill_context = GreenContext(
            num_sms=self.prefill_sms,
            device_id=self.device_id,
            workqueue_concurrency_limit=self.config.greenspace.workqueue_concurrency_limit_prefill,
            priority=self.config.greenspace.stream_priority_prefill,
            enable_hardware_isolation=self.config.greenspace.enable_green_contexts
        )
        self.decode_context = GreenContext(
            num_sms=self.decode_sms,
            device_id=self.device_id,
            workqueue_concurrency_limit=self.config.greenspace.workqueue_concurrency_limit_decode,
            priority=self.config.greenspace.stream_priority_decode,
            enable_hardware_isolation=self.config.greenspace.enable_green_contexts
        )

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
            controller=self.controller
        )
        self.decode_executor = DecodeExecutor(
            config=self.config,
            model=self.model,
            kv_cache=self.kv_cache,
            block_manager=self.block_manager,
            scheduler=self.scheduler,
            decode_context=self.decode_context,
            event_coordinator=self.event_coordinator,
            controller=self.controller
        )

        self._is_started = False
        self.warmup()

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
            self.block_manager.free(warmup_req)
            logger.info("GreenSpace warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup notice: {e}")

    def start(self):
        """Launches the dual concurrent CPU execution loops."""
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
            self.prefill_context.close()
            self.decode_context.close()
            self._is_started = False
            logger.info("Project GreenSpace EngineCore stopped.")

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 128,
        request_id: Optional[str] = None
    ) -> Request:
        """Submits a new inference request to the Prefill queue."""
        req_id = request_id or str(uuid.uuid4())[:8]
        req = Request(request_id=req_id, prompt_token_ids=prompt_token_ids, max_tokens=max_tokens)
        self.scheduler.add_request(req)
        return req

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive system telemetry and latency statistics."""
        step_latencies = self.decode_executor.step_latencies
        rolling_tpot = (
            sum(self.decode_executor._recent_tpots) / len(self.decode_executor._recent_tpots)
            if self.decode_executor._recent_tpots else 0.0
        )
        return {
            "gpu_name": self.gpu_info["name"],
            "total_sms": self.total_sms,
            "prefill_sms": self.prefill_sms,
            "decode_sms": self.decode_sms,
            "prefills_completed": self.prefill_executor.total_prefills_processed,
            "tokens_generated": self.decode_executor.total_tokens_generated,
            "current_rolling_tpot_ms": rolling_tpot,
            "target_tpot_ms": self.config.slo.target_tpot_ms,
            "current_workqueue_limit": self.controller.current_wq_limit,
            "is_elastic_full_sm": self.controller.is_elastic_full_sm,
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


# Pipeline warmup to eliminate cuBLAS jitter
