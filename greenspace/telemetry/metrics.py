"""
Telemetry and Statistical Metrics Tracking for Project GreenSpace.
Calculates TPOT distributions, TTFT, tail percentiles (P50, P90, P99), and jitter reduction.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

from greenspace.core.request import Request


@dataclass
class LatencyReport:
    """Summary of TPOT and TTFT latency statistics."""
    total_requests: int
    total_tokens_generated: int
    mean_tpot_ms: float
    median_tpot_ms: float  # P50
    p90_tpot_ms: float
    p99_tpot_ms: float
    max_tpot_ms: float
    min_tpot_ms: float
    std_tpot_ms: float     # Jitter metric
    mean_ttft_ms: float
    p99_ttft_ms: float
    throughput_tokens_per_sec: float
    tpot_samples: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "mean_tpot_ms": round(self.mean_tpot_ms, 2),
            "median_tpot_ms (P50)": round(self.median_tpot_ms, 2),
            "p90_tpot_ms": round(self.p90_tpot_ms, 2),
            "p99_tpot_ms": round(self.p99_tpot_ms, 2),
            "max_tpot_ms": round(self.max_tpot_ms, 2),
            "min_tpot_ms": round(self.min_tpot_ms, 2),
            "std_tpot_ms (Jitter)": round(self.std_tpot_ms, 2),
            "mean_ttft_ms": round(self.mean_ttft_ms, 2),
            "p99_ttft_ms": round(self.p99_ttft_ms, 2),
            "throughput_tokens_per_sec": round(self.throughput_tokens_per_sec, 2),
        }


class MetricsTracker:
    """Computes high-precision latency percentiles and comparison benchmarks."""
    @staticmethod
    def analyze_requests(requests: List[Request], wall_clock_seconds: float) -> LatencyReport:
        all_tpots = []
        all_ttfts = []
        total_tokens = 0

        for req in requests:
            series = req.get_tpot_series()
            all_tpots.extend(series)
            all_ttfts.append(req.get_ttft())
            total_tokens += req.num_generated_tokens

        if not all_tpots:
            all_tpots = [0.0]
        if not all_ttfts:
            all_ttfts = [0.0]

        tpots = np.array(all_tpots)
        ttfts = np.array(all_ttfts)

        throughput = total_tokens / max(1e-4, wall_clock_seconds)

        return LatencyReport(
            total_requests=len(requests),
            total_tokens_generated=total_tokens,
            mean_tpot_ms=float(np.mean(tpots)),
            median_tpot_ms=float(np.median(tpots)),
            p90_tpot_ms=float(np.percentile(tpots, 90)),
            p99_tpot_ms=float(np.percentile(tpots, 99)),
            max_tpot_ms=float(np.max(tpots)),
            min_tpot_ms=float(np.min(tpots)),
            std_tpot_ms=float(np.std(tpots)),
            mean_ttft_ms=float(np.mean(ttfts)),
            p99_ttft_ms=float(np.percentile(ttfts, 99)),
            throughput_tokens_per_sec=throughput,
            tpot_samples=list(all_tpots)
        )

    @staticmethod
    def compare_reports(baseline: LatencyReport, greenspace: LatencyReport) -> Dict[str, Any]:
        """Calculates relative improvements between baseline and GreenSpace."""
        p99_reduction = (baseline.p99_tpot_ms - greenspace.p99_tpot_ms) / max(1e-4, baseline.p99_tpot_ms) * 100.0
        max_spike_reduction = (baseline.max_tpot_ms - greenspace.max_tpot_ms) / max(1e-4, baseline.max_tpot_ms) * 100.0
        jitter_reduction = (baseline.std_tpot_ms - greenspace.std_tpot_ms) / max(1e-4, baseline.std_tpot_ms) * 100.0

        return {
            "baseline_p99_tpot_ms": baseline.p99_tpot_ms,
            "greenspace_p99_tpot_ms": greenspace.p99_tpot_ms,
            "p99_tpot_reduction_pct": round(p99_reduction, 1),
            "baseline_max_spike_ms": baseline.max_tpot_ms,
            "greenspace_max_spike_ms": greenspace.max_tpot_ms,
            "max_spike_reduction_pct": round(max_spike_reduction, 1),
            "baseline_jitter_std_ms": baseline.std_tpot_ms,
            "greenspace_jitter_std_ms": greenspace.std_tpot_ms,
            "jitter_reduction_pct": round(jitter_reduction, 1),
            "baseline_throughput": baseline.throughput_tokens_per_sec,
            "greenspace_throughput": greenspace.throughput_tokens_per_sec,
        }

