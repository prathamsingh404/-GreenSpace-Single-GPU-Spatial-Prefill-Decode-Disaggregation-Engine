"""
End-to-End Inference and Block Manager lifecycle tests for Project GreenSpace.
"""

import unittest
import time
from greenspace.core.config import EngineConfig, CacheConfig
from greenspace.core.block_manager import BlockManager
from greenspace.core.request import Request, RequestStatus
from greenspace.core.engine import GreenSpaceEngineCore


class TestEngineE2E(unittest.TestCase):
    def test_block_manager_lifecycle(self):
        cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=64)
        bm = BlockManager(cache_cfg)
        self.assertEqual(bm.num_free_blocks, 64)

        # Allocate for a 30-token prompt (requires 2 blocks)
        req = Request("test-req", prompt_token_ids=list(range(30)), max_tokens=10)
        blocks = bm.allocate_for_prompt(req)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(bm.num_free_blocks, 62)

        # Append slot during decode
        req.record_generated_token(100)
        block_id, slot = bm.append_slot_for_decode(req)
        self.assertIn(block_id, blocks)

        # Free blocks
        bm.free(req)
        self.assertEqual(bm.num_free_blocks, 64)
        self.assertEqual(len(req.allocated_blocks), 0)

    def test_engine_inference_e2e(self):
        engine = GreenSpaceEngineCore()
        engine.start()

        req = engine.add_request(prompt_token_ids=[1, 2, 3, 4], max_tokens=5)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 8.0:
            if req.is_finished():
                break
            time.sleep(0.05)

        self.assertTrue(req.is_finished())
        self.assertGreaterEqual(len(req.generated_token_ids), 5)
        self.assertGreater(req.get_mean_tpot(), 0.0)

        telemetry = engine.get_telemetry()
        self.assertGreaterEqual(telemetry["prefills_completed"], 1)
        self.assertGreaterEqual(telemetry["tokens_generated"], 4)

        engine.stop()


if __name__ == "__main__":
    unittest.main()

