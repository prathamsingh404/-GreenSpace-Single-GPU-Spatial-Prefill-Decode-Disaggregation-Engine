"""
Unit tests for Decoupled DualQueueScheduler (Prefill Queue vs Decode Queue).
"""

import unittest
from greenspace.core.config import EngineConfig
from greenspace.scheduler.dual_scheduler import DualQueueScheduler
from greenspace.core.request import Request, RequestStatus


class TestDualQueueScheduler(unittest.TestCase):
    def setUp(self):
        self.config = EngineConfig()
        self.scheduler = DualQueueScheduler(self.config)

    def test_enqueue_prefill(self):
        req1 = Request("req-1", prompt_token_ids=[1, 2, 3], max_tokens=10)
        req2 = Request("req-2", prompt_token_ids=[4, 5, 6, 7], max_tokens=10)
        
        self.scheduler.add_request(req1)
        self.scheduler.add_request(req2)
        
        self.assertEqual(self.scheduler.get_prefill_queue_size(), 2)
        self.assertEqual(self.scheduler.get_decode_queue_size(), 0)

    def test_schedule_prefill_and_handoff(self):
        req = Request("req-handoff", prompt_token_ids=[1, 2], max_tokens=5)
        self.scheduler.add_request(req)
        
        # Schedule next prefill batch
        batch = self.scheduler.schedule_prefills(max_batch_size=4)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].request_id, "req-handoff")
        
        # Handoff to decode queue
        req.status = RequestStatus.READ_ONLY_READY
        self.scheduler.handoff_to_decode(req)
        
        self.assertEqual(self.scheduler.get_prefill_queue_size(), 0)
        self.assertEqual(self.scheduler.get_decode_queue_size(), 1)


if __name__ == "__main__":
    unittest.main()
