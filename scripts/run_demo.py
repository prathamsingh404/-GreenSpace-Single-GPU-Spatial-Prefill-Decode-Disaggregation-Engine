"""
Interactive Demonstration CLI for Project GreenSpace.
Runs a live visual demo comparing GreenSpace vs Monolithic Baseline with Rich formatting.
"""

import time
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from greenspace.core.config import EngineConfig
from greenspace.core.engine import GreenSpaceEngineCore
from greenspace.baseline.monolithic_engine import MonolithicEngine
from greenspace.telemetry.metrics import MetricsTracker

console = Console()


def run_interactive_demo():
    console.print(Panel.fit(
        "[bold green]PROJECT GREENSPACE: SINGLE-GPU SPATIAL PREFILL-DECODE ENGINE[/bold green]\n"
        "[italic cyan]Intra-GPU Hardware Partitioning via CUDA 13.5 Green Contexts & Zero-Copy HBM Handoff[/italic cyan]",
        border_style="green"
    ))

    # Initialize Engine
    with console.status("[bold yellow]Initializing Hardware Green Contexts on GPU...", spinner="dots"):
        config = EngineConfig()
        engine = GreenSpaceEngineCore(config)
        engine.start()

    info = engine.gpu_info
    console.print(f"[bold]Target Hardware:[/bold] {info['name']} (Compute Capability {info['compute_capability'][0]}.{info['compute_capability'][1]})")
    console.print(f"[bold]Spatial Partitioning:[/bold] [red]{engine.prefill_sms} SMs Prefill Zone ({engine.prefill_sms / engine.total_sms:.0%})[/red] | [green]{engine.decode_sms} SMs Decode Zone ({engine.decode_sms / engine.total_sms:.0%})[/green]")
    console.print(f"[bold]Unified Cache Pool:[/bold] {config.cache.num_gpu_blocks} physical blocks in HBM ({config.cache.block_size} tokens/block)\n")

    # Start workload
    console.print("[bold cyan]Submitting Concurrent Workload:[/bold cyan]")
    console.print("  • 4 Continuous Background Decode Streams (Target steady-state decode)")
    console.print("  • A 1024-token prompt burst\n")

    started = time.perf_counter()
    requests = []
    # 4 background decode streams
    for i in range(4):
        req = engine.add_request([100 + i] * 16, max_tokens=25, request_id=f"stream-{i}")
        requests.append(req)

    # Let streams enter decode
    time.sleep(0.5)

    # Inject heavy burst prompts
    console.print("[bold red]>>> INJECTING HEAVY PROMPT BURST (1024 tokens)...[/bold red]")
    b1 = engine.add_request([500] * 1024, max_tokens=15, request_id="heavy-burst-1k")
    requests.append(b1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Generating tokens under burst interference...", total=len(requests))
        
        while not all(r.is_finished() for r in requests):
            finished_count = sum(1 for r in requests if r.is_finished())
            progress.update(task, completed=finished_count)
            time.sleep(0.1)
        progress.update(task, completed=len(requests))

    report = MetricsTracker.analyze_requests(requests, wall_clock_seconds=time.perf_counter() - started)

    # Display Telemetry Results Table
    table = Table(title="Live Inference Telemetry & Latency SLO Report", border_style="cyan")
    table.add_column("Metric", style="bold white")
    table.add_column("Measured Value", style="bold green")
    table.add_column("Target SLO / Threshold", style="yellow")
    table.add_column("Status", style="bold")

    rows = [
        ("Mean TPOT", report.mean_tpot_ms, config.slo.target_tpot_ms),
        ("P50 Median TPOT", report.median_tpot_ms, config.slo.target_tpot_ms),
        ("P99 Tail TPOT", report.p99_tpot_ms, config.slo.slo_threshold_ms),
        ("Maximum token interval", report.max_tpot_ms, config.slo.slo_threshold_ms),
    ]
    for label, value, target in rows:
        status = "[green]PASS[/green]" if value <= target else "[red]MISS[/red]"
        table.add_row(label, f"{value:.2f} ms", f"{target:.2f} ms", status)
    table.add_row("Mean time to first token", f"{report.mean_ttft_ms:.2f} ms", "-", "-")
    table.add_row("Throughput", f"{report.throughput_tokens_per_sec:.2f} tok/s", "-", "-")

    console.print(table)
    engine.stop()
    console.print("\n[bold green]✓ Project GreenSpace execution completed.[/bold green]")


if __name__ == "__main__":
    run_interactive_demo()

