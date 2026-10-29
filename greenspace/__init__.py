"""
Project GreenSpace: Single-GPU Spatial Prefill-Decode Disaggregation Engine.

Achieves intra-GPU spatial disaggregation by physically partitioning a single GPU's
Streaming Multiprocessors (SMs) into isolated Green Contexts for prefill (compute-bound)
and decode (memory-bound) execution phases with zero-copy HBM handoff and closed-loop
PID bandwidth throttling.
"""

__version__ = "1.0.0"
__author__ = "Project GreenSpace Core Team"

from greenspace.core.config import GreenSpaceConfig, EngineConfig, SLOConfig
from greenspace.core.engine import GreenSpaceEngineCore
from greenspace.cuda.green_context import GreenContext, GreenStream

__all__ = [
    "GreenSpaceConfig",
    "EngineConfig",
    "SLOConfig",
    "GreenSpaceEngineCore",
    "GreenContext",
    "GreenStream",
]

