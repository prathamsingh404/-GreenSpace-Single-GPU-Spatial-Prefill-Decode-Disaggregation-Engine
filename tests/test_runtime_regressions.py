"""Regressions for completion, resource exhaustion and decode admission."""

import time

import pytest
from greenspace.core.config import CacheConfig, EngineConfig, ModelConfig
from greenspace.core.engine import GreenSpaceEngineCore
from greenspace.core.request import RequestStatus


def tiny_config():
    return EngineConfig(max_num_seqs=1, cache=CacheConfig(block_size=4, num_gpu_blocks=8),
                        model=ModelConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                                          head_dim=16, intermediate_size=128, vocab_size=128,
                                          num_hidden_layers=1))


def wait_for_terminal(requests):
    deadline = time.monotonic() + 20
    while any(r.status not in (RequestStatus.FINISHED, RequestStatus.FAILED) for r in requests):
        assert time.monotonic() < deadline, requests
        time.sleep(.005)


def test_saturated_queue_and_single_token_completion():
    with GreenSpaceEngineCore(tiny_config()) as engine:
        requests = [engine.add_request([1, 2, 3, 4], max_tokens=(i % 4) + 1) for i in range(64)]
        wait_for_terminal(requests)
        assert all(r.error is None and len(r.generated_token_ids) == r.max_tokens for r in requests)
        assert engine.prefill_executor.total_prefills_processed == len(requests)
        assert engine.block_manager.num_free_blocks == engine.config.cache.num_gpu_blocks


def test_decode_oom_and_duplicate_admission():
    config = tiny_config()
    config.cache.num_gpu_blocks = 1
    engine = GreenSpaceEngineCore(config)
    try:
        request = engine.add_request([1, 2, 3, 4], max_tokens=2, request_id="duplicate")
        with pytest.raises(ValueError, match="Duplicate"):
            engine.add_request([1], request_id="duplicate")
        engine.start()
        wait_for_terminal([request])
        assert request.status == RequestStatus.FAILED and "blocks" in request.error
        followup = engine.add_request([1], max_tokens=2, request_id="duplicate")
        wait_for_terminal([followup])
        assert followup.error is None and len(followup.generated_token_ids) == 2
        assert engine.block_manager.num_free_blocks == 1
    finally:
        engine.stop()
