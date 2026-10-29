"""
Unit tests for Request state transitions, timing telemetry, and lifecycle progression.
"""

import unittest
import time
from greenspace.core.request import Request, RequestStatus


class TestRequestLifecycle(unittest.TestCase):
    def test_initial_state(self):
        req = Request(request_id="req-init", prompt_token_ids=[101, 2054, 2003], max_tokens=16)
        self.assertEqual(req.status, RequestStatus.WAITING)
        self.assertEqual(req.num_prompt_tokens, 3)
        self.assertEqual(len(req.generated_token_ids), 0)
        self.assertFalse(req.is_finished())

    def test_token_generation_and_tpot(self):
        req = Request(request_id="req-gen", prompt_token_ids=[1, 2], max_tokens=3)
        req.status = RequestStatus.DECODING
        
        # Simulate decode token emission
        req.record_generated_token(42)
        time.sleep(0.005)
        req.record_generated_token(43)
        time.sleep(0.005)
        req.record_generated_token(44)

        self.assertEqual(len(req.generated_token_ids), 3)
        self.assertTrue(req.is_finished())
        self.assertGreater(req.get_mean_tpot(), 0.0)

    def test_block_tracking(self):
        req = Request(request_id="req-blocks", prompt_token_ids=list(range(32)), max_tokens=8)
        req.allocated_blocks = [10, 11]
        self.assertEqual(len(req.allocated_blocks), 2)
        self.assertIn(10, req.allocated_blocks)


if __name__ == "__main__":
    unittest.main()
