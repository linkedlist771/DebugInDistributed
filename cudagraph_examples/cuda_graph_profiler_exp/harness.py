#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 20260623
DEFAULT_BATCH = 256
DEFAULT_HIDDEN = 1024
DEFAULT_ITERS = 1
DEFAULT_WARMUP = 25
DEFAULT_BENCH_ITERS = 300


class GraphedToyModel(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.norm(x)
        residual = x
        x = torch.sin(x) * torch.tanh(x)
        reduced = x.mean(dim=-1, keepdim=True)
        x = self.fc3(x + reduced)
        return x + residual


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected on/off boolean, got {value!r}")


def set_seed() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def make_model_and_input(batch: int, hidden: int) -> tuple[nn.Module, torch.Tensor]:
    set_seed()
    torch.set_float32_matmul_precision("high")
    model = GraphedToyModel(hidden).cuda().eval()
    static_input = torch.randn(batch, hidden, device="cuda")
    return model, static_input


def eager_iter(model: nn.Module, static_input: torch.Tensor) -> torch.Tensor:
    return model(static_input)


def make_graph_replay(
    model: nn.Module,
    static_input: torch.Tensor,
    warmup: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    # PyTorch recommends warming up graph-captured work on a side stream.
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(warmup):
            static_output = model(static_input)
    torch.cuda.current_stream().wait_stream(side_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = model(static_input)
    torch.cuda.synchronize()
    return graph, static_output


def run_plain(
    model: nn.Module,
    static_input: torch.Tensor,
    enable_graph: bool,
    warmup: int,
    iters: int,
) -> None:
    if enable_graph:
        graph, _ = make_graph_replay(model, static_input, warmup)
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        for idx in range(iters):
            torch.cuda.nvtx.range_push(f"PLAIN_ITER_{idx}")
            with torch.profiler.record_function(f"PLAIN_ITER_{idx}"):
                graph.replay()
                torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
    else:
        for _ in range(warmup):
            eager_iter(model, static_input)
        torch.cuda.synchronize()
        for idx in range(iters):
            torch.cuda.nvtx.range_push(f"PLAIN_ITER_{idx}")
            with torch.profiler.record_function(f"PLAIN_ITER_{idx}"):
                eager_iter(model, static_input)
                torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


def run_torch_profiler(
    model: nn.Module,
    static_input: torch.Tensor,
    enable_graph: bool,
    warmup: int,
    iters: int,
    trace_path: Path,
    profile_scope: str,
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    if enable_graph and profile_scope == "capture":
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(warmup):
                model(static_input)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            with_modules=True,
        ) as prof:
            with torch.profiler.record_function("PROFILE_ITER_0"):
                with torch.cuda.graph(graph):
                    model(static_input)
            torch.cuda.synchronize()
            prof.step()

        prof.export_chrome_trace(str(trace_path))
        return

    if enable_graph:
        graph, _ = make_graph_replay(model, static_input, warmup)

        def run_one(idx: int) -> None:
            with torch.profiler.record_function(f"PROFILE_ITER_{idx}"):
                graph.replay()

    else:
        for _ in range(warmup):
            eager_iter(model, static_input)
        torch.cuda.synchronize()

        def run_one(idx: int) -> None:
            with torch.profiler.record_function(f"PROFILE_ITER_{idx}"):
                eager_iter(model, static_input)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        with_modules=True,
    ) as prof:
        for idx in range(iters):
            run_one(idx)
            torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(str(trace_path))


def measure_latency(
    model: nn.Module,
    static_input: torch.Tensor,
    enable_graph: bool,
    warmup: int,
    bench_iters: int,
) -> dict[str, float | int | bool]:
    if enable_graph:
        graph, _ = make_graph_replay(model, static_input, warmup)
        for _ in range(warmup):
            graph.replay()
    else:
        for _ in range(warmup):
            eager_iter(model, static_input)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if enable_graph:
            graph.replay()
        else:
            eager_iter(model, static_input)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    return {
        "graph": enable_graph,
        "warmup": warmup,
        "bench_iters": bench_iters,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.fmean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p90_ms": statistics.quantiles(times_ms, n=10, method="inclusive")[8],
    }


def write_metadata(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "seed": SEED,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "args": serializable_args,
        "pid": os.getpid(),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA Graph profiler visibility harness")
    parser.add_argument("--graph", type=parse_bool, default=False)
    parser.add_argument("--profiler", type=parse_bool, default=False)
    parser.add_argument("--profile-scope", choices=["replay", "capture"], default="replay")
    parser.add_argument("--trace-path", type=Path, default=Path("trace.json"))
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--mode", choices=["run", "bench"], default="run")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--bench-iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--latency-json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    model, static_input = make_model_and_input(args.batch, args.hidden)
    if args.metadata_path is not None:
        write_metadata(args.metadata_path, args)

    if args.mode == "bench":
        result = measure_latency(
            model=model,
            static_input=static_input,
            enable_graph=args.graph,
            warmup=args.warmup,
            bench_iters=args.bench_iters,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.latency_json is not None:
            args.latency_json.parent.mkdir(parents=True, exist_ok=True)
            args.latency_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return

    if args.profiler:
        run_torch_profiler(
            model=model,
            static_input=static_input,
            enable_graph=args.graph,
            warmup=args.warmup,
            iters=args.iters,
            trace_path=args.trace_path,
            profile_scope=args.profile_scope,
        )
    else:
        run_plain(
            model=model,
            static_input=static_input,
            enable_graph=args.graph,
            warmup=args.warmup,
            iters=args.iters,
        )


if __name__ == "__main__":
    main()
