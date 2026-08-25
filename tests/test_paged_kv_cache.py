"""Unit tests for PagedKVCache allocation and indexing."""
import unittest
import torch
from greenspace.core.config import CacheConfig, ModelConfig
from greenspace.core.paged_kv_cache import PagedKVCache

class TestPagedKVCache(unittest.TestCase):
    def test_cache_allocation(self):
        m_cfg = ModelConfig(num_layers=2, num_kv_heads=2, hidden_dim=64)
        c_cfg = CacheConfig(block_size=16, num_gpu_blocks=8)
        cache = PagedKVCache(c_cfg, m_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(len(cache.k_cache), 2)

if __name__ == "__main__":
    unittest.main()
