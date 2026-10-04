"""
Closed-Loop PID Latency Controller & Elastic SM Rebalancer.
Continuously monitors rolling TPOT telemetry, computes error against the SLO target,
and dynamically throttles the Prefill Green Context workqueue concurrency limit
to prevent memory bandwidth contention and L2 cache thrashing.
"""

import time
import logging
from typing import Optional, List, Dict, Any
from greenspace.core.config import SLOConfig, GreenSpaceConfig
from greenspace.cuda.green_context import GreenContext

logger = logging.getLogger("GreenSpace.PIDController")


class ClosedLoopPIDController:
    """
    PID Controller for bandwidth throttling and SM elasticity:
      e(t) = TPOT_measured - TPOT_target
      u(t) = Kp * e(t) + Ki * integral(e) + Kd * derivative(e)
    Controls:
      - workqueue_concurrency_limit on Prefill Green Context
      - prefill chunking / submission pacing
      - elastic SM re-allocation between prefill and decode
    """
    def __init__(
        self,
        slo_config: SLOConfig,
        greenspace_config: GreenSpaceConfig,
        prefill_context: GreenContext,
        decode_context: GreenContext,
        total_gpu_sms: int = 20
    ):
        self.slo_config = slo_config
        self.greenspace_config = greenspace_config
        self.prefill_context = prefill_context
        self.decode_context = decode_context
        self.total_gpu_sms = total_gpu_sms

        # PID parameters
        self.kp = slo_config.pid_kp
        self.ki = slo_config.pid_ki
        self.kd = slo_config.pid_kd
        self.target_tpot = slo_config.target_tpot_ms
        self.slo_threshold = slo_config.slo_threshold_ms

        # PID state variables
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.last_update_time = time.perf_counter()

        # Actuator state
        self.current_wq_limit = greenspace_config.workqueue_concurrency_limit_prefill
        self.current_chunk_size = slo_config.max_chunk_size
        self.is_elastic_full_sm = False

        # Telemetry history for auditing and benchmark reports
        self.history_records: List[Dict[str, Any]] = []

    def update_telemetry(self, rolling_tpot_ms: float, num_active_decodes: int, step: int):
        """
        Called every N steps with the latest measured rolling TPOT.
        Computes PID error and applies hardware workqueue throttle.
        """
        now = time.perf_counter()
        dt = max(1e-4, now - self.last_update_time)
        self.last_update_time = now

        # Handle elastic SM and workqueue rebalancing
        if self.greenspace_config.enable_dynamic_rebinding:
            if num_active_decodes == 0 and not self.is_elastic_full_sm:
                # Idle decode: Give maximum workqueue and compute throughput to prefill
                self.is_elastic_full_sm = True
                self.prefill_context.set_workqueue_concurrency_limit(self.slo_config.max_workqueue_concurrency)
                logger.debug(f"[Elastic Balance] Decode idle. Expanded Prefill workqueue limit to {self.slo_config.max_workqueue_concurrency}.")
            elif num_active_decodes > 0 and self.is_elastic_full_sm:
                # Active decode: Restore isolated nominal workqueue limits
                self.is_elastic_full_sm = False
                self.prefill_context.set_workqueue_concurrency_limit(self.greenspace_config.workqueue_concurrency_limit_prefill)
                logger.debug(f"[Elastic Balance] Decode active. Restored Prefill workqueue limit to {self.greenspace_config.workqueue_concurrency_limit_prefill}.")

        # If no active decodes, no TPOT to control
        if num_active_decodes == 0 or rolling_tpot_ms <= 0.0:
            return

        # Calculate error: positive error means TPOT is too high (SLO breached)
        error = rolling_tpot_ms - self.target_tpot
        self.integral_error = max(-50.0, min(50.0, self.integral_error + error * dt))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        # Control output
        control_output = (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative)

        # Actuate: When error > 0, throttle workqueue concurrency and chunk size
        nominal_wq = self.greenspace_config.workqueue_concurrency_limit_prefill
        new_wq = int(round(nominal_wq - control_output))
        new_wq = max(self.slo_config.min_workqueue_concurrency, min(self.slo_config.max_workqueue_concurrency, new_wq))

        # Chunk size scaling
        if error > (self.slo_threshold - self.target_tpot):
            # Aggressive throttling for severe violations
            new_chunk = max(self.slo_config.min_chunk_size, self.current_chunk_size // 2)
        elif error < 0:
            # Recovery
            new_chunk = min(self.slo_config.max_chunk_size, self.current_chunk_size + 128)
        else:
            new_chunk = self.current_chunk_size

        if new_wq != self.current_wq_limit:
            self.current_wq_limit = new_wq
            self.prefill_context.set_workqueue_concurrency_limit(new_wq)

        self.current_chunk_size = new_chunk

        # Record telemetry sample
        self.history_records.append({
            "timestamp": now,
            "step": step,
            "measured_tpot_ms": rolling_tpot_ms,
            "target_tpot_ms": self.target_tpot,
            "error_ms": error,
            "control_output": control_output,
            "workqueue_concurrency_limit": self.current_wq_limit,
            "prefill_chunk_size": self.current_chunk_size,
            "is_elastic_full_sm": self.is_elastic_full_sm,
            "num_active_decodes": num_active_decodes,
        })


# Dynamic WQ limit actuation
