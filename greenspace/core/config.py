"""
Configuration dataclasses for Project GreenSpace:
Engine, Hardware Partitioning, Model, Cache, and Closed-Loop Latency SLO.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class GreenSpaceConfig:
    """Hardware partitioning and Green Context configuration."""
    # SM Partition ratio: 0.6 = 60% prefill, 0.4 = 40% decode
    sm_partition_prefill: float = 0.6
    sm_partition_decode: float = 0.4
    
    device_id: int = 0
    enable_green_contexts: bool = True
    enable_dynamic_rebinding: bool = True  # Re-allocate SMs to prefill when decode queue is idle
    
    # Workqueue concurrency limits for memory bandwidth control
    workqueue_concurrency_limit_prefill: int = 8
    workqueue_concurrency_limit_decode: int = 32

    # CUDA Stream priority: negative numbers indicate higher priority in CUDA
    stream_priority_prefill: int = 0
    stream_priority_decode: int = -1

    def get_sm_counts(self, total_sms: int, alignment: int = 8) -> Tuple[int, int]:
        """Calculates aligned SM counts for prefill and decode partitions."""
        raw_prefill = int(round(total_sms * self.sm_partition_prefill))
        raw_decode = total_sms - raw_prefill

        # Ensure at least minimum partition size per zone
        min_sms = max(1, alignment)
        if total_sms >= min_sms * 2:
            prefill_sms = max(min_sms, (raw_prefill // alignment) * alignment)
            decode_sms = max(min_sms, total_sms - prefill_sms)
        else:
            prefill_sms = max(1, raw_prefill)
            decode_sms = max(1, raw_decode)

        return prefill_sms, decode_sms


@dataclass
class ModelConfig:
    """LLM Architecture specification for forward passes and memory footprint."""
    model_name: str = "greenspace-llama-1b"
    hidden_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA (Grouped Query Attention)
    head_dim: int = 128
    intermediate_size: int = 5632
    vocab_size: int = 32000
    max_position_embeddings: int = 8192
    dtype: torch.dtype = torch.float16
    use_synthetic_weights: bool = True


@dataclass
class CacheConfig:
    """Unified GPU HBM Paged KV Cache memory pool configuration."""
    block_size: int = 16  # Number of tokens stored per physical block
    num_gpu_blocks: int = 1024  # Total allocatable blocks in GPU HBM
    gpu_memory_fraction: float = 0.8
    enable_zero_copy_handoff: bool = True


@dataclass
class SLOConfig:
    """Service-Level Objective (SLO) & PID controller configuration for TPOT stability."""
    target_tpot_ms: float = 20.0  # Target Time-Per-Output-Token (ms)
    slo_threshold_ms: float = 25.0  # Violation threshold triggering aggressive throttling
    telemetry_window_steps: int = 5  # Step frequency for PID evaluation
    
    # PID gains
    pid_kp: float = 0.08  # Proportional gain
    pid_ki: float = 0.015 # Integral gain
    pid_kd: float = 0.01  # Derivative gain
    
    # Actuator limits
    min_workqueue_concurrency: int = 1
    max_workqueue_concurrency: int = 16
    min_chunk_size: int = 64
    max_chunk_size: int = 2048


@dataclass
class EngineConfig:
    """Root configuration for GreenSpace Engine."""
    greenspace: GreenSpaceConfig = field(default_factory=GreenSpaceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    slo: SLOConfig = field(default_factory=SLOConfig)
    max_num_seqs: int = 64
    max_model_len: int = 8192

