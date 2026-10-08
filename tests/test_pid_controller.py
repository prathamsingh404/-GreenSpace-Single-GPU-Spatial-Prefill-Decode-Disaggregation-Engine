"""
Unit tests for the Closed-Loop PID Latency Controller.
"""

import unittest
from greenspace.core.config import SLOConfig, GreenSpaceConfig
from greenspace.cuda.green_context import GreenContext
from greenspace.core.controller import ClosedLoopPIDController


class TestPIDController(unittest.TestCase):
    def setUp(self):
        self.slo_cfg = SLOConfig(target_tpot_ms=25.0, slo_threshold_ms=30.0, pid_kp=0.1, pid_ki=0.01, pid_kd=0.01)
        self.gs_cfg = GreenSpaceConfig(workqueue_concurrency_limit_prefill=16)
        self.prefill_ctx = GreenContext(num_sms=8, enable_hardware_isolation=False)
        self.decode_ctx = GreenContext(num_sms=8, enable_hardware_isolation=False)
        self.controller = ClosedLoopPIDController(
            self.slo_cfg, self.gs_cfg, self.prefill_ctx, self.decode_ctx, total_gpu_sms=20
        )

    def tearDown(self):
        self.prefill_ctx.close()
        self.decode_ctx.close()

    def test_nominal_state(self):
        # When TPOT matches target, concurrency limit should remain steady
        self.controller.update_telemetry(rolling_tpot_ms=25.0, num_active_decodes=2, step=1)
        self.assertEqual(self.controller.current_wq_limit, 16)

    def test_throttling_on_violation(self):
        # Latency spike to 45ms (20ms above target)
        self.controller.update_telemetry(rolling_tpot_ms=45.0, num_active_decodes=2, step=2)
        # Should throttle workqueue limit below 16
        self.assertLess(self.controller.current_wq_limit, 16)
        self.assertGreaterEqual(self.controller.current_wq_limit, self.slo_cfg.min_workqueue_concurrency)

    def test_elastic_rebalancing(self):
        # When 0 active decodes, elasticity expands capacity
        self.controller.update_telemetry(rolling_tpot_ms=0.0, num_active_decodes=0, step=3)
        self.assertTrue(self.controller.is_elastic_full_sm)

        # When active decode returns, restores partition
        self.controller.update_telemetry(rolling_tpot_ms=24.0, num_active_decodes=1, step=4)
        self.assertFalse(self.controller.is_elastic_full_sm)


if __name__ == "__main__":
    unittest.main()

