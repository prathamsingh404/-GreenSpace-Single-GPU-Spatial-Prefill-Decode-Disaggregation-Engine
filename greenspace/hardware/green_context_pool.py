"""Pre-provisioned complementary partitions for a single CUDA device."""

from dataclasses import dataclass
import math

import torch

from greenspace.cuda.driver import driver
from greenspace.cuda.green_context import GreenContext, GreenStream


@dataclass(frozen=True)
class ContextPair:
    prefill: GreenContext
    decode: GreenContext
    prefill_stream: GreenStream
    decode_stream: GreenStream


class GreenContextLattice:
    """Selecting a pair never creates a CUDA context or stream."""

    def __init__(self, device_id: int = 0, ratios=(.2, .3, .4, .5, .6), *,
                 prefill_limit: int | None = None, decode_limit: int | None = None,
                 prefill_priority: int = 0, decode_priority: int = -1):
        info = driver.get_device_info(device_id)
        self.total_sms = info["sm_count"]
        self._contexts: list[GreenContext] = []
        self.pairs: dict[int, ContextPair] = {}
        self.full_gpu_stream = torch.cuda.Stream(device=device_id)
        try:
            for ratio in ratios:
                if not 0 < ratio < 1:
                    raise ValueError("Decode ratio must lie strictly between zero and one")
                # Split the smaller side. The driver assigns all residual SMs to its complement.
                split_decode = ratio <= .5
                target = self.total_sms * (ratio if split_decode else 1 - ratio)
                quantum = info["sm_coscheduled_alignment"] or 1
                count = max(info["min_sm_partition_size"] or 1,
                            math.floor(target / quantum + .5) * quantum)
                remainder, group = driver.split(device_id, count)
                p, d = (remainder, group) if split_decode else (group, remainder)
                if d.sm.smCount in self.pairs:
                    continue
                pc = GreenContext(p.sm.smCount, device_id, prefill_limit, prefill_priority, resource=p)
                self._contexts.append(pc)
                dc = GreenContext(d.sm.smCount, device_id, decode_limit, decode_priority, resource=d)
                self._contexts.append(dc)
                self.pairs[dc.num_sms] = ContextPair(pc, dc, pc.Stream(), dc.Stream())
            if not self.pairs:
                raise ValueError("Provide at least one decode ratio")
            self._by_count = [self.pairs[min(self.pairs, key=lambda size: abs(size - count))]
                              for count in range(self.total_sms + 1)]
        except BaseException:
            self.close()
            raise

    def get_pair(self, decode_sm_ratio: float) -> ContextPair:
        if not self.pairs:
            raise RuntimeError("The lattice is closed")
        if not 0 < decode_sm_ratio < 1:
            raise ValueError("Decode ratio must lie strictly between zero and one")
        return self._by_count[round(decode_sm_ratio * self.total_sms)]

    def close(self):
        while self._contexts:
            self._contexts[-1].close()
            self._contexts.pop()
        self.pairs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
