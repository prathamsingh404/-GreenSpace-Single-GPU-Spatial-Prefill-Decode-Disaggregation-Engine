"""
Ablation Study: Quantifying the impact of each GreenSpace pillar.
Configurations tested:
  1. Baseline Monolithic (No spatial partition, single thread)
  2. Dual-Thread Multi-Stream (Independent streams, but shared monolithic SMs)
  3. Green Context Hardware Slicing (Spatial SM isolation, fixed workqueues)
  4. Full GreenSpace (Spatial SM isolation + Closed-Loop PID Bandwidth Controller)
"""

import os
import json
import time
import logging
from typing import Dict, Any

from greenspace.core.config import EngineConfig, GreenSpaceConfig, SLOConfig
from greenspace.core.engine import GreenSpaceEngineCore
from greenspace.baseline.monolithic_engine import MonolithicEngine
from greenspace.telemetry.metrics import MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GreenSpace.Ablation")


def run_ablation():
    logger.info("Executing Project GreenSpace Architectural Ablation Study...")
    results: Dict[str, Any] = {}

    # Config 1: Monolithic Baseline
    logger.info("[1/4] Running Monolithic Baseline...")
    cfg1 = EngineConfig()
    eng1 = MonolithicEngine(cfg1)
    # Warmup
    eng1.add_request([1, 2], max_tokens=2)
    eng1.step()
    # Benchmark
    r1 = eng1.add_request([10] * 512, max_tokens=15)
    r2 = eng1.add_request([20] * 16, max_tokens=15)
    for _ in range(30):
        eng1.step()
        if r1.is_finished() and r2.is_finished():
            break
    rep1 = MetricsTracker.analyze_requests([r1, r2], wall_clock_seconds=1.0)
    results["1_monolithic_baseline"] = rep1.to_dict()

    # Config 2: Dual Stream without Green Context (Fallback Mode)
    logger.info("[2/4] Running Dual-Stream without Hardware SM Isolation...")
    cfg2 = EngineConfig()
    cfg2.greenspace.enable_green_contexts = False
    cfg2.greenspace.enable_dynamic_rebinding = False
    eng2 = GreenSpaceEngineCore(cfg2)
    eng2.start()
    r1 = eng2.add_request([10] * 512, max_tokens=15)
    r2 = eng2.add_request([20] * 16, max_tokens=15)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10.0:
        if r1.is_finished() and r2.is_finished():
            break
        time.sleep(0.05)
    rep2 = MetricsTracker.analyze_requests([r1, r2], wall_clock_seconds=time.perf_counter() - t0)
    eng2.stop()
    results["2_dual_stream_unpartitioned"] = rep2.to_dict()

    # Config 3: Green Context Hardware SM Partitioning (No PID)
    logger.info("[3/4] Running Green Context Spatial Isolation without PID...")
    cfg3 = EngineConfig()
    cfg3.greenspace.enable_green_contexts = True
    cfg3.greenspace.enable_dynamic_rebinding = False
    eng3 = GreenSpaceEngineCore(cfg3)
    eng3.controller = None  # Disable PID controller updates
    eng3.start()
    r1 = eng3.add_request([10] * 512, max_tokens=15)
    r2 = eng3.add_request([20] * 16, max_tokens=15)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10.0:
        if r1.is_finished() and r2.is_finished():
            break
        time.sleep(0.05)
    rep3 = MetricsTracker.analyze_requests([r1, r2], wall_clock_seconds=time.perf_counter() - t0)
    eng3.stop()
    results["3_green_context_only"] = rep3.to_dict()

    # Config 4: Full GreenSpace Engine (Green Context + PID Bandwidth Throttling)
    logger.info("[4/4] Running Full GreenSpace (Green Context + PID Bandwidth Control)...")
    cfg4 = EngineConfig()
    cfg4.greenspace.enable_green_contexts = True
    cfg4.greenspace.enable_dynamic_rebinding = True
    eng4 = GreenSpaceEngineCore(cfg4)
    eng4.start()
    r1 = eng4.add_request([10] * 512, max_tokens=15)
    r2 = eng4.add_request([20] * 16, max_tokens=15)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10.0:
        if r1.is_finished() and r2.is_finished():
            break
        time.sleep(0.05)
    rep4 = MetricsTracker.analyze_requests([r1, r2], wall_clock_seconds=time.perf_counter() - t0)
    eng4.stop()
    results["4_full_greenspace"] = rep4.to_dict()

    logger.info("=" * 70)
    logger.info("ABLATION STUDY SUMMARY (P99 TPOT & JITTER COMPARISON):")
    for k, v in results.items():
        logger.info(f"{k:32s} | P99 TPOT: {v['p99_tpot_ms']} ms | Jitter (Std): {v['std_tpot_ms (Jitter)']} ms")
    logger.info("=" * 70)

    os.makedirs("benchmarks/results", exist_ok=True)
    with open("benchmarks/results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_ablation()

