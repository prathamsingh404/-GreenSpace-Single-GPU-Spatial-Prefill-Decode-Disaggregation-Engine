"""
Unified GPU HBM Paged Key-Value Cache Pool for Zero-Copy Disaggregation.
Both Prefill and Decode Green Contexts directly access these physical blocks in shared HBM.
"""

from typing import List, Tuple
import torch
from greenspace.core.config import CacheConfig, ModelConfig


class PagedKVCache:
    """
    Physical KV cache tensor pool resident in GPU High-Bandwidth Memory (HBM).
    Layout: [num_layers, 2 (key/val), num_blocks, num_kv_heads, block_size, head_dim].
    """
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, device: str = "cuda"):
        self.model_config = model_config
        self.cache_config = cache_config
        self.device = device
        
        self.num_layers = model_config.num_hidden_layers
        self.num_blocks = cache_config.num_gpu_blocks
        self.block_size = cache_config.block_size
        self.num_kv_heads = model_config.num_key_value_heads
        self.head_dim = model_config.head_dim
        self.dtype = model_config.dtype

        # Allocate physical key and value tensors in GPU HBM
        # Shape: [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        self.key_cache = torch.zeros(
            (self.num_layers, self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )
        self.val_cache = torch.zeros(
            (self.num_layers, self.num_blocks, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=self.dtype,
            device=self.device
        )

    def write_prompt_kv(
        self,
        layer_idx: int,
        block_ids: List[int],
        keys: torch.Tensor,
        values: torch.Tensor
    ):
        """
        Written by Prefill Thread: Writes prompt key and value projections into allocated
        physical blocks in HBM.
        keys, values: [seq_len, num_kv_heads, head_dim]
        """
        seq_len = keys.shape[0]
        for token_idx in range(seq_len):
            block_idx = block_ids[token_idx // self.block_size]
            slot_idx = token_idx % self.block_size
            self.key_cache[layer_idx, block_idx, :, slot_idx, :] = keys[token_idx]
            self.val_cache[layer_idx, block_idx, :, slot_idx, :] = values[token_idx]

    def write_token_kv(
        self,
        layer_idx: int,
        block_id: int,
        slot_idx: int,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        """
        Written by Decode Thread: Writes a single new token's key and value projection
        into the designated block and slot.
        key, value: [num_kv_heads, head_dim]
        """
        self.key_cache[layer_idx, block_id, :, slot_idx, :] = key
        self.val_cache[layer_idx, block_id, :, slot_idx, :] = value

    def get_kv_for_blocks(
        self,
        layer_idx: int,
        block_ids: List[int],
        total_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read by Decode Thread: Gathers cached keys and values across the request's
        allocated blocks for attention computation.
        Returns:
            keys: [total_tokens, num_kv_heads, head_dim]
            values: [total_tokens, num_kv_heads, head_dim]
        """
        if not block_ids or total_tokens <= 0:
            empty = torch.empty((0, self.num_kv_heads, self.head_dim), dtype=self.dtype, device=self.device)
            return empty, empty

        # Gather the physical blocks
        k_blocks = self.key_cache[layer_idx, block_ids]  # [num_blocks, num_kv_heads, block_size, head_dim]
        v_blocks = self.val_cache[layer_idx, block_ids]

        # Reshape to continuous sequence: [num_blocks * block_size, num_kv_heads, head_dim]
        num_blocks = len(block_ids)
        k_seq = k_blocks.permute(0, 2, 1, 3).reshape(num_blocks * self.block_size, self.num_kv_heads, self.head_dim)
        v_seq = v_blocks.permute(0, 2, 1, 3).reshape(num_blocks * self.block_size, self.num_kv_heads, self.head_dim)

        # Slice exact valid tokens up to total_tokens
        return k_seq[:total_tokens], v_seq[:total_tokens]

    def memory_footprint_bytes(self) -> int:
        """Returns total memory allocated for the KV cache pool in bytes."""
        return (self.key_cache.nelement() + self.val_cache.nelement()) * self.key_cache.element_size()


# Vectorized prompt write helper

# Single token decode slot write

# Zero copy block gathering
