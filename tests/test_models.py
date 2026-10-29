"""
Unit tests for Transformer architecture layers: RMSNorm, MLP, and GreenSpaceTransformerModel.
"""

import unittest
import torch
from greenspace.core.config import ModelConfig, CacheConfig
from greenspace.models.transformer import RMSNorm, MLP, GreenSpaceTransformerModel
from greenspace.core.paged_kv_cache import PagedKVCache


class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dim = 128
        self.model_cfg = ModelConfig(
            vocab_size=1000,
            hidden_size=self.dim,
            intermediate_size=self.dim * 2,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            dtype=torch.float32
        )
        self.cache_cfg = CacheConfig(block_size=16, num_gpu_blocks=16)

    def test_rmsnorm_forward(self):
        norm = RMSNorm(hidden_size=self.dim, dtype=torch.float32).to(self.device)
        x = torch.randn(2, 8, self.dim, device=self.device)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        var = torch.mean(out ** 2, dim=-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

    def test_mlp_forward(self):
        mlp = MLP(self.model_cfg).to(self.device)
        x = torch.randn(2, self.dim, device=self.device)
        out = mlp(x)
        self.assertEqual(out.shape, x.shape)

    def test_model_prefill_and_decode(self):
        model = GreenSpaceTransformerModel(self.model_cfg, device=self.device)
        cache = PagedKVCache(self.model_cfg, self.cache_cfg, device=self.device)
        
        # Test prefill pass
        prompt_tokens = [10, 20, 30, 40]
        block_ids = [0]
        logits = model.forward_prefill(prompt_tokens, block_ids, cache)
        self.assertEqual(logits.shape, (1, self.model_cfg.vocab_size))

        # Test decode pass
        next_token = 50
        decode_logits = model.forward_decode(
            token_id=next_token,
            block_ids=block_ids,
            total_tokens=5,
            target_block_id=0,
            target_slot_idx=4,
            kv_cache=cache
        )
        self.assertEqual(decode_logits.shape, (1, self.model_cfg.vocab_size))


if __name__ == "__main__":
    unittest.main()
