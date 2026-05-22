"""Inference Request Lifecycle representation."""
from enum import Enum

class RequestStatus(Enum):
    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    READ_ONLY_READY = "READ_ONLY_READY"
    DECODING = "DECODING"
    FINISHED = "FINISHED"
