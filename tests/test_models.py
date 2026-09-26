"""Unit tests for Transformer and PagedAttention modules."""
import unittest
import torch
from greenspace.core.config import ModelConfig
from greenspace.models.transformer import RMSNorm

class TestModels(unittest.TestCase):
    def test_rmsnorm(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 4, 64)
        out = norm(x)
        self.assertEqual(out.shape, (2, 4, 64))

if __name__ == "__main__":
    unittest.main()
