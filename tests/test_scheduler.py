"""Unit tests for Decoupled DualQueueScheduler."""
import unittest
from greenspace.core.config import EngineConfig
from greenspace.scheduler.dual_scheduler import DualQueueScheduler
from greenspace.core.request import Request

class TestScheduler(unittest.TestCase):
    def test_scheduling(self):
        cfg = EngineConfig()
        sched = DualQueueScheduler(cfg)
        req = Request("s1", [1, 2], 5)
        sched.add_request(req)
        self.assertEqual(sched.get_prefill_queue_size(), 1)

if __name__ == "__main__":
    unittest.main()
