"""
Unit tests for Green Context creation, stream allocation, and workqueue limits.
"""

import unittest
import torch
from greenspace.cuda.driver import driver
from greenspace.cuda.green_context import GreenContext, GreenStream


class TestGreenContext(unittest.TestCase):
    def setUp(self):
        self.device_id = 0
        self.gpu_info = driver.get_device_info(self.device_id)

    def test_device_info(self):
        self.assertGreater(self.gpu_info["sm_count"], 0)
        self.assertIn("name", self.gpu_info)
        self.assertGreater(self.gpu_info["l2_cache_bytes"], 0)

    def test_green_context_lifecycle(self):
        ctx = GreenContext(
            num_sms=8,
            device_id=self.device_id,
            workqueue_concurrency_limit=16
        )
        self.assertIsNotNone(ctx)

        # Create stream
        stream = ctx.Stream()
        self.assertIsInstance(stream, GreenStream)

        # Run kernel on stream
        with torch.cuda.stream(stream.torch_stream):
            a = torch.randn(256, 256, device="cuda")
            b = torch.randn(256, 256, device="cuda")
            c = torch.matmul(a, b)
        stream.synchronize()

        self.assertEqual(c.shape, (256, 256))

        # Test workqueue limit update
        ctx.set_workqueue_concurrency_limit(4)
        self.assertEqual(ctx.workqueue_concurrency_limit, 4)

        ctx.close()


if __name__ == "__main__":
    unittest.main()


# SM partition validation tests
