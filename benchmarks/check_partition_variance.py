"""Repeat the CP1 GEMM/GEMV comparison with alternating order and primed events."""

import argparse
import json
from pathlib import Path
from statistics import variance

import torch

from greenspace.hardware.green_context_pool import GreenContextLattice
from greenspace.hardware.topology_probe import manifest


def sample(streams, matrix, vector, out_p, out_d, steps=20):
    prefill, decode = streams
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(steps)]
    # CUDA creates timing events on their first record. Keep this outside the measurement.
    for start, end in events:
        start.record(decode)
        end.record(decode)
    decode.synchronize()
    for start, end in events:
        with torch.cuda.stream(prefill):
            torch.mm(matrix, matrix, out=out_p)
        with torch.cuda.stream(decode):
            start.record()
            torch.mm(vector, matrix, out=out_d)
            end.record()
    prefill.synchronize()
    decode.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def compare(trials=6):
    if trials < 2 or trials % 2:
        raise ValueError("Use an even number of trials to balance measurement order")
    matrix = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    vector = matrix[:1].clone()
    buffers = matrix, vector, torch.empty_like(matrix), torch.empty_like(vector)
    torch.cuda.current_stream().synchronize()
    results = []
    with GreenContextLattice(prefill_limit=8, decode_limit=32) as lattice:
        pair = lattice.get_pair(.4)
        streams = {"shared": (torch.cuda.Stream(), torch.cuda.Stream(priority=-1)),
                   "spatial": (pair.prefill_stream.torch_stream, pair.decode_stream.torch_stream)}
        for stream_pair in streams.values():
            sample(stream_pair, *buffers, steps=10)
        for trial in range(trials):
            order = ["shared", "spatial"] if trial % 2 == 0 else ["spatial", "shared"]
            row = {"order": order}
            for mode in order:
                times = sample(streams[mode], *buffers)
                row[mode] = {"samples_ms": times, "variance_ms2": variance(times)}
            row["spatial_lower"] = row["spatial"]["variance_ms2"] < row["shared"]["variance_ms2"]
            results.append(row)
    return {"hardware": manifest(), "trials": results,
            "spatial_lower_count": sum(row["spatial_lower"] for row in results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoint_1_variance_trials.json"))
    args = parser.parse_args()
    result = compare(args.trials)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Spatial variance was lower in {result['spatial_lower_count']} of {args.trials} paired trials")
