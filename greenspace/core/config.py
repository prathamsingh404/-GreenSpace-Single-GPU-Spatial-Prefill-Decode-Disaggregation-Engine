"""GreenSpace Engine Configuration Dataclasses."""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GreenSpaceConfig:
    prefill_sm_fraction: float = 0.60
    decode_sm_fraction: float = 0.40
    device_id: int = 0
