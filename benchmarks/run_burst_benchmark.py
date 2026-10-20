"""
Burst-Traffic Benchmark Suite: Project GreenSpace vs Baseline Monolithic Engine.
Simulates bursty LLM traffic with concurrent decode streams interrupted by heavy prompt bursts,
measuring per-token TPOT spikes, P99 latency, jitter, and throughput.
"""

import os
import json
import time
import argparse
import logging
from typing import List, Dict, Any
import numpy as np

from greenspace.core.config import EngineConfig, GreenSpaceConfig, SLOConfig, ModelConfig
from greenspace.core.engine import GreenSpaceEngineCore
from greenspace.baseline.monolithic_engine import MonolithicEngine
from greenspace.telemetry.metrics import MetricsTracker, LatencyReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("GreenSpace.Benchmark")


def run_workload_on_greenspace(
    num_background_streams: int = 4,
    background_tokens: int = 25,
    burst_prompts: List[int] = [512, 1024, 2048],
    burst_step: int = 8
) -> LatencyReport:
    """Executes burst workload on GreenSpace Engine with spatial Green Contexts and PID."""
    logger.info(">>> Running Benchmark on Project GreenSpace EngineCore...")
    config = EngineConfig()
    engine = GreenSpaceEngineCore(config)
    engine.start()

    requests = []
    # 1. Start background decode streams
    for i in range(num_background_streams):
        req = engine.add_request(
            prompt_token_ids=[100 + i] * 16,
            max_tokens=background_tokens,
            request_id=f"bg-stream-{i}"
        )
        requests.append(req)

    # Allow background streams to enter decode phase
    time.sleep(0.5)

    # 2. Inject heavy burst prompts while background streams are actively decoding
    logger.info(f"Injecting {len(burst_prompts)} heavy prompt bursts...")
    for idx, prompt_len in enumerate(burst_prompts):
        burst_req = engine.add_request(
            prompt_token_ids=[200 + idx] * prompt_len,
            max_tokens=15,
            request_id=f"heavy-burst-{idx}-{prompt_len}tok"
        )
        requests.append(burst_req)
        time.sleep(0.1)

    # 3. Wait for all requests to complete
    t_start = time.perf_counter()
    timeout = 30.0
    while time.perf_counter() - t_start < timeout:
        if all(r.is_finished() for r in requests):
            break
        time.sleep(0.05)

    wall_clock = time.perf_counter() - t_start
    report = MetricsTracker.analyze_requests(requests, wall_clock)
    engine.stop()
    return report


def run_workload_on_monolithic(
    num_background_streams: int = 4,
    background_tokens: int = 25,
    burst_prompts: List[int] = [512, 1024, 2048],
    burst_step: int = 8
) -> LatencyReport:
    """Executes burst workload on Baseline Monolithic Engine without spatial isolation."""
    logger.info(">>> Running Benchmark on Baseline Monolithic Engine...")
    config = EngineConfig()
    engine = MonolithicEngine(config)

    requests = []
    for i in range(num_background_streams):
        req = engine.add_request(
            prompt_token_ids=[100 + i] * 16,
            max_tokens=background_tokens,
            request_id=f"bg-stream-{i}"
        )
        requests.append(req)

    t_start = time.perf_counter()
    step_idx = 0
    burst_injected = False

    while any(not r.is_finished() for r in requests) and step_idx < 100:
        # Inject bursts at specific step
        if step_idx == burst_step and not burst_injected:
            logger.info(f"Injecting {len(burst_prompts)} heavy prompt bursts into monolithic queue...")
            for idx, prompt_len in enumerate(burst_prompts):
                burst_req = engine.add_request(
                    prompt_token_ids=[200 + idx] * prompt_len,
                    max_tokens=15,
                    request_id=f"heavy-burst-{idx}-{prompt_len}tok"
                )
                requests.append(burst_req)
            burst_injected = True

        engine.step()
        step_idx += 1

    wall_clock = time.perf_counter() - t_start
    report = MetricsTracker.analyze_requests(requests, wall_clock)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Project GreenSpace vs Monolithic Engine Burst Benchmark")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    burst_prompts = [256, 512, 1024]
    num_streams = 4
    tokens_per_stream = 20

    logger.info("=" * 70)
    logger.info("PROJECT GREENSPACE: SINGLE-GPU SPATIAL PREFILL-DECODE BENCHMARK")
    logger.info("=" * 70)

    # Run Monolithic Baseline
    monolithic_report = run_workload_on_monolithic(
        num_background_streams=num_streams,
        background_tokens=tokens_per_stream,
        burst_prompts=burst_prompts
    )

    # Run GreenSpace Engine
    greenspace_report = run_workload_on_greenspace(
        num_background_streams=num_streams,
        background_tokens=tokens_per_stream,
        burst_prompts=burst_prompts
    )

    # Compare
    comparison = MetricsTracker.compare_reports(monolithic_report, greenspace_report)

    logger.info("-" * 70)
    logger.info("BENCHMARK COMPARISON RESULTS:")
    logger.info(f"Baseline P99 TPOT:   {monolithic_report.p99_tpot_ms:.2f} ms")
    logger.info(f"GreenSpace P99 TPOT: {greenspace_report.p99_tpot_ms:.2f} ms (Reduction: {comparison['p99_tpot_reduction_pct']}%)")
    logger.info(f"Baseline Max Spike:  {monolithic_report.max_tpot_ms:.2f} ms")
    logger.info(f"GreenSpace Max Spike:{greenspace_report.max_tpot_ms:.2f} ms (Reduction: {comparison['max_spike_reduction_pct']}%)")
    logger.info(f"Baseline Jitter Std: {monolithic_report.std_tpot_ms:.2f} ms")
    logger.info(f"GreenSpace Jitter Std:{greenspace_report.std_tpot_ms:.2f} ms (Reduction: {comparison['jitter_reduction_pct']}%)")
    logger.info("-" * 70)

    # Save to JSON
    output_data = {
        "baseline": monolithic_report.to_dict(),
        "greenspace": greenspace_report.to_dict(),
        "comparison": comparison,
        "raw_samples": {
            "baseline_tpot": monolithic_report.tpot_samples,
            "greenspace_tpot": greenspace_report.tpot_samples
        }
    }
    json_path = os.path.join(args.output_dir, "burst_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()

