"""
case02b: 带【完整 Python 调用栈】+【CUDA 设备侧算子】的 profiler 对比。

产出三类文件：
  1. trace_eager_vs_graph.json      -> 拖进 https://ui.perfetto.dev
       - GPU 时间线上是真实 kernel；点任一 CPU 算子，右侧 args 里能看到完整 python 调用栈
       - 连接 CPU launch -> GPU kernel 的箭头(flow events)说明谁启动了哪个 kernel
  2. stacks_cuda.txt / stacks_cpu.txt -> folded stacks，给 FlameGraph 用
       flamegraph.pl stacks_cuda.txt > cuda.svg   # https://github.com/brendangregg/FlameGraph
  3. 终端打印按 CUDA 时间排序、并带调用栈分组的算子表
"""

import torch
import torch.nn as nn
from typing import Callable
from loguru import logger
from torch.profiler import profile, ProfilerActivity, record_function

DEV = "cuda"


def warmup_on_side_stream(fn: Callable, iters: int = 3):
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):          # 小写 stream = 上下文管理器
        for _ in range(iters):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)


class MLP(nn.Module):
    def __init__(self, d: int = 4096, hidden: int = 16384):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def case02_profile_with_stacks(batch: int = 64, d: int = 4096, iters: int = 20,
                               trace_path: str = "trace_eager_vs_graph.json"):
    logger.critical("Profile (with full python stack + CUDA ops): eager vs graph replay")

    model = MLP(d=d).to(DEV).eval()
    static_in = torch.randn(batch, d, device=DEV)

    def fwd():
        with torch.no_grad():
            return model(static_in)

    # 1) 侧流预热
    warmup_on_side_stream(fwd, iters=3)

    # 2) 捕获
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = fwd()

    # 3) eager 也先预热，避免 lazy init 进 profiler
    with torch.no_grad():
        for _ in range(3):
            _ = model(static_in)
    g.replay()
    torch.cuda.synchronize()

    # 4) profiling —— 关键是这几个开关
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,      # 记录输入 shape
        profile_memory=True,     # 记录显存分配
        with_stack=True,         # ★ 记录完整 python / C++ 调用栈
        with_modules=True,       # ★ 记录 nn.Module 层级(看得到 MLP.fc1 等)
        with_flops=True,         # 估算 FLOPs
    ) as prof:

        with record_function("eager"):
            with torch.no_grad():
                for _ in range(iters):
                    _ = model(static_in)
            torch.cuda.synchronize()

        with record_function("cudagraph_replay"):
            for _ in range(iters):
                g.replay()
            torch.cuda.synchronize()

    # 5a) chrome/perfetto trace：含 GPU kernel + flow + 每个算子的调用栈(在 args 里)
    prof.export_chrome_trace(trace_path)
    logger.debug(f"perfetto trace: {trace_path}")

    # 5b) folded stacks -> 火焰图。两个指标各导一份
    #     注意：export_stacks 依赖 with_stack=True
    prof.export_stacks("stacks_cuda.txt", "self_cuda_time_total")
    prof.export_stacks("stacks_cpu.txt", "self_cpu_time_total")
    logger.debug("folded stacks: stacks_cuda.txt / stacks_cpu.txt  "
                 "(flamegraph.pl stacks_cuda.txt > cuda.svg)")

    # 5c) 终端表：按 CUDA 时间排序，并把每个算子折叠到最近 N 层调用栈
    logger.debug(
        "\n=== group by stack (CUDA time) ===\n"
        + prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cuda_time_total", row_limit=20
        )
    )
    logger.debug(
        "\n=== group by input shape ===\n"
        + prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total", row_limit=15
        )
    )

    ref = fwd()
    torch.cuda.synchronize()
    logger.debug(f"replay vs eager 最大误差: {(static_out - ref).abs().max().item():.3e}")


if __name__ == "__main__":
    case02_profile_with_stacks()