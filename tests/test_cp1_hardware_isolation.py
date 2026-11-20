"""Hardware acceptance tests. Run on an idle GPU with CUPTI and NVRTC available."""

from collections import defaultdict
import gzip
import os
import json
from pathlib import Path
from statistics import variance

import pytest
import torch

from greenspace.hardware.isolation_probe import verify_pairs
from greenspace.hardware.green_context_pool import GreenContextLattice
from greenspace.hardware.topology_probe import manifest


@pytest.fixture(scope="module")
def report():
    directory = Path(os.environ.get("GREENSPACE_ARTIFACT_DIR", "artifacts"))
    directory.mkdir(parents=True, exist_ok=True)
    data = {"hardware": manifest(), "checks": {}}
    yield data
    checks = data["checks"]
    data["passed"] = len(checks) == 3 and all(value["passed"] for value in checks.values())
    (directory / "checkpoint_1_report.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_physical_smids(report):
    with GreenContextLattice(prefill_limit=8, decode_limit=32) as lattice:
        rows = verify_pairs(lattice, 0)
    result = report["checks"]["physical_smids"] = {"pairs": rows, "passed": all(row["passed"] for row in rows)}
    assert result["passed"], rows


def timed_workload(p, d, matrix, vector, out_p, out_d, steps):
    samples = []
    for _ in range(steps):
        with torch.cuda.stream(p):
            torch.mm(matrix, matrix, out=out_p)
        with torch.cuda.stream(d):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            torch.mm(vector, matrix, out=out_d)
            end.record()
            samples.append((start, end))
    p.synchronize()
    d.synchronize()
    return [start.elapsed_time(end) for start, end in samples]


def test_concurrent_latency_and_trace(report, tmp_path):
    matrix = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    vector = matrix[:1].clone()
    out_p, out_d = torch.empty_like(matrix), torch.empty_like(vector)
    torch.cuda.current_stream().synchronize()
    with GreenContextLattice(prefill_limit=8, decode_limit=32) as lattice:
        pair = lattice.get_pair(.4)
        spatial = pair.prefill_stream.torch_stream, pair.decode_stream.torch_stream
        shared = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
        args = matrix, vector, out_p, out_d
        for streams in (shared, spatial):
            timed_workload(*streams, *args, 3)
        shared_ms = timed_workload(*shared, *args, 20)
        spatial_ms = timed_workload(*spatial, *args, 20)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profile:
            timed_workload(*spatial, *args, 20)
        trace = tmp_path / "overlap.json"
        profile.export_chrome_trace(str(trace))
        trace_data = json.loads(trace.read_text(encoding="utf-8"))
        trace_data.pop("host_name", None)
        trace_data.pop("traceName", None)
        artifact_dir = Path(os.environ.get("GREENSPACE_ARTIFACT_DIR", "artifacts"))
        with gzip.open(artifact_dir / "checkpoint_1_overlap_trace.json.gz", "wt", encoding="utf-8") as output:
            json.dump(trace_data, output, separators=(",", ":"))
        intervals = defaultdict(list)
        for event in trace_data["traceEvents"]:
            if event.get("cat") == "kernel":
                intervals[event["args"]["stream"]].append((event["ts"], event["ts"] + event["dur"]))
        overlap = 0.0
        if len(intervals) == 2:
            first, second = intervals.values()
            overlap = sum(max(0, min(a1, b1) - max(a0, b0))
                          for a0, a1 in first for b0, b1 in second)
        result = report["checks"]["concurrency"] = {
            "trace": "checkpoint_1_overlap_trace.json.gz",
            "shared_decode_ms": shared_ms, "spatial_decode_ms": spatial_ms,
            "shared_variance_ms2": variance(shared_ms), "spatial_variance_ms2": variance(spatial_ms),
            "kernel_counts_by_stream": {str(k): len(v) for k, v in intervals.items()},
            "kernel_overlap_us": overlap, "passed": overlap > 0 and variance(spatial_ms) < variance(shared_ms)}
        assert result["passed"], result


def test_lifecycle(report):
    before = torch.cuda.memory_allocated()
    for _ in range(5):
        with GreenContextLattice() as lattice:
            pair = lattice.get_pair(.4)
            assert pair is lattice.get_pair(.4)
            with torch.cuda.stream(pair.decode_stream.torch_stream):
                value = torch.ones(1, device="cuda")
                assert value.item() == 1
            del value
        assert pair.prefill.closed and pair.decode.closed
    after = torch.cuda.memory_allocated()
    report["checks"]["lifecycle"] = dict(cycles=5, before_bytes=before, after_bytes=after, passed=after == before)
    assert after == before
