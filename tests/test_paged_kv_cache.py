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
            num_hidden_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32
        )
        self.cache_cfg = CacheConfig(
            block_size=16,
            num_gpu_blocks=32
        )
        self.cache = PagedKVCache(self.model_cfg, self.cache_cfg, device=self.device)

    def test_cache_structure(self):
        expected_shape = (
            self.model_cfg.num_hidden_layers,
            self.cache_cfg.num_gpu_blocks,
            self.model_cfg.num_key_value_heads,
            self.cache_cfg.block_size,
            self.model_cfg.head_dim
        )
        self.assertEqual(self.cache.key_cache.shape, expected_shape)
        self.assertEqual(self.cache.val_cache.shape, expected_shape)

    def test_write_and_gather_token_kv(self):
        layer_idx = 0
        block_id = 2
        slot_idx = 5
        
        k_val = torch.ones(self.model_cfg.num_key_value_heads, self.model_cfg.head_dim, dtype=self.model_cfg.dtype, device=self.device) * 3.14
        v_val = torch.ones(self.model_cfg.num_key_value_heads, self.model_cfg.head_dim, dtype=self.model_cfg.dtype, device=self.device) * 2.71
        
        self.cache.write_token_kv(layer_idx, block_id, slot_idx, k_val, v_val)
        
        stored_k = self.cache.key_cache[layer_idx, block_id, :, slot_idx, :]
        stored_v = self.cache.val_cache[layer_idx, block_id, :, slot_idx, :]
        self.assertTrue(torch.allclose(stored_k, k_val, atol=1e-2))
        self.assertTrue(torch.allclose(stored_v, v_val, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
