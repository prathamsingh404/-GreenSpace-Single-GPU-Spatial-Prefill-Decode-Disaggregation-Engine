"""
Full Transformer Decoder Architecture (Llama-style with RMSNorm, SwiGLU, and PagedAttention).
Supports decoupled Prefill forward pass and Decode forward pass on isolated Green Context streams.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from greenspace.core.config import ModelConfig
from greenspace.core.paged_kv_cache import PagedKVCache
from greenspace.models.attention import PagedAttention


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MLP(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=config.dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerDecoderLayer(nn.Module):
    """Single Transformer layer with PagedAttention and SwiGLU MLP."""
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = PagedAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, dtype=config.dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, dtype=config.dtype)

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        block_ids: List[int],
        kv_cache: PagedKVCache
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_prefill(hidden_states, block_ids, kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_decode(
        self,
        hidden_state: torch.Tensor,
        block_ids: List[int],
        total_tokens: int,
        target_block_id: int,
        target_slot_idx: int,
        kv_cache: PagedKVCache
    ) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        hidden_state = self.self_attn.forward_decode(
            hidden_state, block_ids, total_tokens, target_block_id, target_slot_idx, kv_cache
        )
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state
        return hidden_state


class GreenSpaceTransformerModel(nn.Module):
    """Full LLM Engine Model executed across dual Green Context zones."""
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, dtype=config.dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)

        # Initialize weights
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward_prefill(
        self,
        prompt_token_ids: List[int],
        block_ids: List[int],
        kv_cache: PagedKVCache
    ) -> torch.Tensor:
        """
        Executes dense prompt forward pass on Prefill Green Context.
        Returns next-token logits: [1, vocab_size]
        """
        input_tensor = torch.tensor(prompt_token_ids, dtype=torch.long, device=self.device)
        hidden_states = self.embed_tokens(input_tensor)

        for layer in self.layers:
            hidden_states = layer.forward_prefill(hidden_states, block_ids, kv_cache)

        hidden_states = self.norm(hidden_states)
        # Take the last token's representation for next-token prediction
        last_hidden = hidden_states[-1:]
        logits = self.lm_head(last_hidden)
        return logits

    def forward_decode(
        self,
        token_id: int,
        block_ids: List[int],
        total_tokens: int,
        target_block_id: int,
        target_slot_idx: int,
        kv_cache: PagedKVCache
    ) -> torch.Tensor:
        """
        Executes single-token autoregressive step on Decode Green Context.
        Returns next-token logits: [1, vocab_size]
        """
        input_tensor = torch.tensor([token_id], dtype=torch.long, device=self.device)
        hidden_state = self.embed_tokens(input_tensor)

        for layer in self.layers:
            hidden_state = layer.forward_decode(
                hidden_state, block_ids, total_tokens, target_block_id, target_slot_idx, kv_cache
            )

        hidden_state = self.norm(hidden_state)
        logits = self.lm_head(hidden_state)
        return logits


# SwiGLU projection layers

# TransformerDecoderLayer assembly
