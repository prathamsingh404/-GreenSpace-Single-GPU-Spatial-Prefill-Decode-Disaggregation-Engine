"""
Unit tests for GPU Memory Barrier coherence and EventCoordinator zero-copy handoffs.
"""

import unittest
import torch
from greenspace.cuda.green_context import GreenContext
from greenspace.cuda.memory_barrier import GPUMemoryBarrier
from greenspace.cuda.event_coordinator import EventCoordinator
from greenspace.core.request import Request, RequestStatus


class TestZeroCopyHandoff(unittest.TestCase):
    def setUp(self):
        self.device_id = 0
        self.prefill_ctx = GreenContext(num_sms=8, device_id=self.device_id)
        self.decode_ctx = GreenContext(num_sms=8, device_id=self.device_id)
        self.s_prefill = self.prefill_ctx.Stream()
        self.s_decode = self.decode_ctx.Stream()
        self.coordinator = EventCoordinator(device_id=self.device_id)

    def tearDown(self):
        self.prefill_ctx.close()
        self.decode_ctx.close()

    def test_memory_barrier_coherence(self):
        # Verify bit-exact coherence between producer stream and consumer stream without host copy
        is_coherent, latency_us = GPUMemoryBarrier.verify_coherence(
            producer_stream=self.s_prefill.torch_stream,
            consumer_stream=self.s_decode.torch_stream,
            tensor_size=1048576
        )
        self.assertTrue(is_coherent, "GPU memory barrier failed coherence verification!")
        self.assertGreaterEqual(latency_us, 0.0)

    def test_event_coordinator_lifecycle(self):
        req = Request(request_id="test-req-1", prompt_token_ids=[1, 2, 3, 4], max_tokens=10)
        self.assertEqual(req.status, RequestStatus.WAITING)

        # Prefill finishes
        event = self.coordinator.signal_prefill_completed(req, self.s_prefill)
        self.assertEqual(req.status, RequestStatus.READ_ONLY_READY)
        self.assertIsNotNone(req.prefill_event)

        # Decode acquires
        self.coordinator.acquire_for_decode(req, self.s_decode)
        self.assertEqual(req.status, RequestStatus.DECODING)

        # Measure handoff latency
        handoff_ms = self.coordinator.get_handoff_latency_ms(req)
        self.assertGreaterEqual(handoff_ms, 0.0)

        # Release
        self.coordinator.release_request(req)
        self.assertIsNone(req.prefill_event)


if __name__ == "__main__":
    unittest.main()

