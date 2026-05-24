"""Inference Request Lifecycle representation."""
from enum import Enum

class RequestStatus(Enum):
    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    READ_ONLY_READY = "READ_ONLY_READY"
    DECODING = "DECODING"
    FINISHED = "FINISHED"

import time
from typing import List, Optional

class Request:
    def __init__(self, request_id: str, prompt_token_ids: List[int], max_tokens: int = 64):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.max_tokens = max_tokens
        self.status = RequestStatus.WAITING
        self.generated_token_ids: List[int] = []
        self.allocated_blocks: List[int] = []
