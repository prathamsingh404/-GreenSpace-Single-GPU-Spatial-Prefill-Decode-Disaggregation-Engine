"""
Unit tests for Transformer architecture layers: RMSNorm, SwiGLU, and PagedAttention.
"""

import unittest
import torch
from greenspace.core.config import ModelConfig, CacheConfig
from greenspace.models.transformer import RMSNorm, SwiGLUMLP, GreenSpaceTransformerModel
from greenspace.core.paged_kv_cache import PagedKVCache


class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dim = 64
        self.model_cfg = ModelConfig(
            vocab_size=1000,
            hidden_dim=self.dim,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=128
        )
        self.cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=16)

    def test_rmsnorm_forward(self):
        norm = RMSNorm(dim=self.dim).to(self.device)
        x = torch.randn(2, 8, self.dim, device=self.device)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        # Verify output variance is normalized
        var = torch.mean(out ** 2, dim=-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

    def test_swiglu_mlp_forward(self):
        mlp = SwiGLUMLP(hidden_dim=self.dim, intermediate_dim=self.dim * 2).to(self.device)
        x = torch.randn(2, 4, self.dim, device=self.device)
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

    def test_model_prefill_and_decode(self):
        model = GreenSpaceTransformerModel(self.model_cfg).to(self.device)
        cache = PagedKVCache(self.cache_cfg, self.model_cfg, device=self.device)
        
        # Test prefill pass
        prompt_tokens = torch.tensor([[10, 20, 30, 40]], dtype=torch.long, device=self.device)
        block_tables = [[0]]
        logits = model.forward_prefill(prompt_tokens, cache, block_tables)
        self.assertEqual(logits.shape, (1, 4, self.model_cfg.vocab_size))

        # Test decode pass
        next_token = torch.tensor([[50]], dtype=torch.long, device=self.device)
        seq_lens = torch.tensor([5], dtype=torch.int32, device=self.device)
        decode_logits = model.forward_decode(next_token, cache, block_tables, seq_lens)
        self.assertEqual(decode_logits.shape, (1, 1, self.model_cfg.vocab_size))


if __name__ == "__main__":
    unittest.main()
