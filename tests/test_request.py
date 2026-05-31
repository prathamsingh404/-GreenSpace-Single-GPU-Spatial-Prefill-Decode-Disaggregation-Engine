"""Unit tests for Request state transitions."""
import unittest
from greenspace.core.request import Request, RequestStatus

class TestRequest(unittest.TestCase):
    def test_lifecycle(self):
        req = Request("r1", [1, 2, 3], max_tokens=5)
        self.assertEqual(req.status, RequestStatus.WAITING)
        req.record_generated_token(10)
        self.assertEqual(req.generated_token_ids, [10])

if __name__ == "__main__":
    unittest.main()

    def test_mean_tpot(self):
        req = Request("r2", [1], max_tokens=2)
        req.record_generated_token(10)
        self.assertGreaterEqual(req.get_mean_tpot(), 0.0)
