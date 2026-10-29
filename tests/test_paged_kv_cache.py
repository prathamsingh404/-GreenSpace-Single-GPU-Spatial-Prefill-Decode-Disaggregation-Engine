"""
Unit tests for PagedKVCache allocation, indexing, and block gathering.
"""

import unittest
import torch
from greenspace.core.config import CacheConfig, ModelConfig
from greenspace.core.paged_kv_cache import PagedKVCache


class TestPagedKVCachePool(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_cfg = ModelConfig(
            num_layers=2,
            hidden_dim=128,
            num_heads=4,
            num_kv_heads=2
        )
        self.cache_cfg = CacheConfig(
            block_size=16,
            num_gpu_blocks=32
        )
        self.cache = PagedKVCache(self.cache_cfg, self.model_cfg, device=self.device)

    def test_cache_structure(self):
        self.assertEqual(len(self.cache.k_cache), self.model_cfg.num_layers)
        self.assertEqual(len(self.cache.v_cache), self.model_cfg.num_layers)
        
        # Verify block tensor shapes: [num_blocks, num_kv_heads, block_size, head_dim]
        head_dim = self.model_cfg.hidden_dim // self.model_cfg.num_heads
        expected_shape = (
            self.cache_cfg.num_gpu_blocks,
            self.model_cfg.num_kv_heads,
            self.cache_cfg.block_size,
            head_dim
        )
        self.assertEqual(self.cache.k_cache[0].shape, expected_shape)

    def test_write_and_gather_token_kv(self):
        layer_idx = 0
        block_id = 2
        slot_idx = 5
        head_dim = self.model_cfg.hidden_dim // self.model_cfg.num_heads
        
        k_val = torch.ones(self.model_cfg.num_kv_heads, head_dim, device=self.device) * 3.14
        v_val = torch.ones(self.model_cfg.num_kv_heads, head_dim, device=self.device) * 2.71
        
        self.cache.write_token_kv(layer_idx, block_id, slot_idx, k_val, v_val)
        
        # Verify values stored
        stored_k = self.cache.k_cache[layer_idx][block_id, :, slot_idx, :]
        stored_v = self.cache.v_cache[layer_idx][block_id, :, slot_idx, :]
        self.assertTrue(torch.allclose(stored_k, k_val))
        self.assertTrue(torch.allclose(stored_v, v_val))


if __name__ == "__main__":
    unittest.main()
