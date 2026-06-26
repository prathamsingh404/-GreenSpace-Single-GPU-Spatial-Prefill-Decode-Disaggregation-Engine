"""Unit tests for CUDA Driver and Green Context subsystem."""
import unittest
from greenspace.cuda.driver import driver

class TestGreenContext(unittest.TestCase):
    def setUp(self):
        self.device_id = 0
        self.gpu_info = driver.get_device_info(self.device_id)

    def test_device_info(self):
        self.assertGreater(self.gpu_info["sm_count"], 0)
        self.assertIn("name", self.gpu_info)

if __name__ == "__main__":
    unittest.main()

    def test_l2_cache(self):
        self.assertGreater(self.gpu_info["l2_cache_bytes"], 0)
