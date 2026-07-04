"""High-level GreenContext wrapper."""
import torch
from typing import Optional

class GreenStream:
    def __init__(self, raw_handle: int, device_id: int = 0):
        self.raw_handle = raw_handle
        self.device_id = device_id
        self._torch_stream = torch.cuda.Stream(device=device_id)

    @property
    def torch_stream(self):
        return self._torch_stream

        if raw_handle:
            try:
                self._torch_stream = torch.cuda.ExternalStream(raw_handle, device=device_id)
            except Exception:
                pass
