"""GreenSpace Engine Configuration Dataclasses."""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GreenSpaceConfig:
    prefill_sm_fraction: float = 0.60
    decode_sm_fraction: float = 0.40
    device_id: int = 0

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_layers: int = 8
    num_heads: int = 16
    num_kv_heads: int = 4
    max_seq_len: int = 4096

@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 256
