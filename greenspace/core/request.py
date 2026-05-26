"""
Request and Sequence state representations for Project GreenSpace.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Any
import time
import torch


class RequestStatus(Enum):
    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    READ_ONLY_READY = "READ_ONLY_READY"  # Zero-copy handoff point: KV cache written, event recorded
    DECODING = "DECODING"
    FINISHED = "FINISHED"


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 128
    stop_tokens: List[int] = field(default_factory=list)


class Request:
    """Represents a single generative inference request through the engine pipeline."""
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        max_tokens: int = 128,
        sampling_params: Optional[SamplingParams] = None,
        arrival_time: Optional[float] = None
    ):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_tokens = max_tokens
        self.sampling_params = sampling_params or SamplingParams(max_tokens=max_tokens)
        
        self.status = RequestStatus.WAITING
        self.allocated_blocks: List[int] = []
        self.generated_token_ids: List[int] = []
        
        # Timing telemetry
        self.arrival_time = arrival_time or time.perf_counter()
        self.prefill_start_time: Optional[float] = None
        self.prefill_end_time: Optional[float] = None
        self.decode_start_time: Optional[float] = None
        self.finish_time: Optional[float] = None
        
        # Timestamps of each generated token for exact per-token TPOT calculation
        self.token_timestamps: List[float] = []

        # Synchronization event
        self.prefill_event: Optional[torch.cuda.Event] = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_generated_tokens

    def record_generated_token(self, token_id: int):
        now = time.perf_counter()
        self.generated_token_ids.append(token_id)
        self.token_timestamps.append(now)

    def is_finished(self) -> bool:
        if len(self.generated_token_ids) >= self.max_tokens:
            return True
        if self.sampling_params.stop_tokens and self.generated_token_ids:
            if self.generated_token_ids[-1] in self.sampling_params.stop_tokens:
                return True
        return False

    def get_ttft(self) -> float:
        """Time To First Token (TTFT) in milliseconds."""
        if not self.token_timestamps:
            if self.prefill_end_time and self.arrival_time:
                return (self.prefill_end_time - self.arrival_time) * 1000.0
            return 0.0
        return (self.token_timestamps[0] - self.arrival_time) * 1000.0

    def get_tpot_series(self) -> List[float]:
        """Returns the series of Time-Per-Output-Token (TPOT) latencies in milliseconds."""
        if len(self.token_timestamps) < 2:
            return []
        series = []
        for i in range(1, len(self.token_timestamps)):
            diff = (self.token_timestamps[i] - self.token_timestamps[i - 1]) * 1000.0
            series.append(diff)
        return series

    def get_mean_tpot(self) -> float:
        """Mean TPOT in milliseconds."""
        series = self.get_tpot_series()
        return sum(series) / len(series) if series else 0.0

    def __repr__(self) -> str:
        return (
            f"<Request id={self.request_id} status={self.status.name} "
            f"prompt_len={self.num_prompt_tokens} gen_len={self.num_generated_tokens} "
            f"blocks={len(self.allocated_blocks)}>"
        )

