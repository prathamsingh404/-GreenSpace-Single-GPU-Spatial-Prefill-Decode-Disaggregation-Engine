"""
Unit tests for Decoupled DualQueueScheduler (Prefill Queue vs Decode Queue).
"""

import unittest
from greenspace.scheduler.dual_scheduler import DualQueueScheduler
from greenspace.core.request import Request, RequestStatus


class TestDualQueueScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = DualQueueScheduler(max_num_seqs=32)

    def test_enqueue_prefill(self):
        req1 = Request("req-1", prompt_token_ids=[1, 2, 3], max_tokens=10)
        req2 = Request("req-2", prompt_token_ids=[4, 5, 6, 7], max_tokens=10)
        
        self.scheduler.add_request(req1)
        self.scheduler.add_request(req2)
        
        self.assertEqual(self.scheduler.num_pending_prefills, 2)
        self.assertEqual(self.scheduler.num_active_decodes, 0)

    def test_schedule_prefill_and_handoff(self):
        req = Request("req-handoff", prompt_token_ids=[1, 2], max_tokens=5)
        self.scheduler.add_request(req)
        
        # Pull next prefill request
        prefill_req = self.scheduler.get_next_prefill_request()
        self.assertIsNotNone(prefill_req)
        self.assertEqual(prefill_req.request_id, "req-handoff")
        self.assertEqual(prefill_req.status, RequestStatus.PREFILLING)
        
        # Handoff to decode pool
        prefill_req.status = RequestStatus.READ_ONLY_READY
        self.scheduler.transfer_to_decode(prefill_req)
        
        self.assertEqual(self.scheduler.num_pending_prefills, 0)
        self.assertEqual(self.scheduler.num_active_decodes, 1)

        # Pull active decodes
        batch = self.scheduler.get_active_decode_batch()
        self.assertEqual(len(batch), 1)

        # Mark finished
        self.scheduler.mark_finished(prefill_req)
        self.assertEqual(self.scheduler.num_active_decodes, 0)
        self.assertEqual(prefill_req.status, RequestStatus.FINISHED)


if __name__ == "__main__":
    unittest.main()
