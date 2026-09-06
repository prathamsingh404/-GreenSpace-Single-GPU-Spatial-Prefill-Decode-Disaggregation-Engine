"""
Paged Grouped-Query Attention (GQA) for Prefill (compute-bound) and Decode (memory-bound).
Directly reads and writes to the unified GPU HBM PagedKVCache pool.
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from greenspace.core.config import ModelConfig
from greenspace.core.paged_kv_cache import PagedKVCache


class PagedAttention(nn.Module):
    """
    Implements Paged Attention with Grouped Query Attention (GQA).
    Decoupled into:
      1) compute_prefill: dense matrix multiplication, writes prompt KV to paged cache.
      2) compute_decode: memory-bound single-token query against cached pages.
    """
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=config.dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=config.dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=config.dtype)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, dtype=config.dtype)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        block_ids: List[int],
        kv_cache: PagedKVCache
    ) -> torch.Tensor:
        """
        Executed on Prefill Green Context (compute-bound dense phase).
        hidden_states: [seq_len, hidden_size]
        """
        seq_len, _ = hidden_states.shape

        # Dense Q, K, V projections saturating Tensor Cores
        q = self.q_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_len, self.num_kv_heads, self.head_dim)

        # Write prompt KV directly to shared HBM PagedKVCache blocks
        kv_cache.write_prompt_kv(self.layer_idx, block_ids, k, v)

        # Expand KV heads for GQA: [seq_len, num_heads, head_dim]
        k_expanded = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_expanded = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Transpose for batch-wise multi-head attention: [num_heads, seq_len, head_dim]
        q_t = q.transpose(0, 1)
        k_t = k_expanded.transpose(0, 1)
        v_t = v_expanded.transpose(0, 1)

        # Compute scaled dot-product attention with causal mask
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device), diagonal=1)
        attn_scores = attn_scores + causal_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_output = torch.matmul(attn_weights, v_t)  # [num_heads, seq_len, head_dim]

        # Reshape to [seq_len, hidden_size]
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)

    def forward_decode(
        self,
        hidden_state: torch.Tensor,
        block_ids: List[int],
        total_tokens: int,
        target_block_id: int,
        target_slot_idx: int,
        kv_cache: PagedKVCache
    ) -> torch.Tensor:
        """
        Executed on Decode Green Context (memory-bound autoregressive phase).
        hidden_state: [1, hidden_size]
        """
        # Single-token projections
        q = self.q_proj(hidden_state).view(1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_state).view(1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_state).view(1, self.num_kv_heads, self.head_dim)

        # Append new token KV to paged cache
        kv_cache.write_token_kv(self.layer_idx, target_block_id, target_slot_idx, k.squeeze(0), v.squeeze(0))

        # Gather all past keys and values from physical HBM blocks (Zero-Copy)
        k_all, v_all = kv_cache.get_kv_for_blocks(self.layer_idx, block_ids, total_tokens)

        # Expand KV heads for GQA: [total_tokens, num_heads, head_dim]
        k_all_expanded = k_all.repeat_interleave(self.num_queries_per_kv, dim=1)
        v_all_expanded = v_all.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Attention: Q is [1, num_heads, head_dim], K is [total_tokens, num_heads, head_dim]
        q_t = q.transpose(0, 1)                       # [num_heads, 1, head_dim]
        k_t = k_all_expanded.transpose(0, 1)          # [num_heads, total_tokens, head_dim]
        v_t = v_all_expanded.transpose(0, 1)          # [num_heads, total_tokens, head_dim]

        # Compute dot product across all historical tokens
        attn_scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale  # [num_heads, 1, total_tokens]
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(hidden_state.dtype)
        attn_output = torch.matmul(attn_weights, v_t)  # [num_heads, 1, head_dim]

        attn_output = attn_output.transpose(0, 1).reshape(1, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


# Specialized prefill forward pass

# Specialized decode forward pass
