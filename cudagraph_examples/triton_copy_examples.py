# -*- coding: utf-8 -*-
"""
单测: 用 lru_cache 的进程级单例 a_gpu, MLP 层内零侵入获取, 不需要在
model forward 的每一层传参。

结构(模拟真实推理):
  - get_a_gpu(): lru_cache 单例 → 每次返回同一个 tensor → 地址固定, 可被 capture
  - set_num_rows(A): 只在 runner 层(replay 之外)每 step 调一次
  - MLP.forward(): 内部 get_a_gpu() 拿指针传给 triton kernel
  - 整个 Model(NUM_LAYERS 层) forward 被 capture 进一个 graph
  - replay 覆盖 A < / == / > A_capture 以及 A=0
  - 另测 eager 模式走同一套代码路径也正确
"""

from functools import lru_cache

import torch
import torch.nn as nn
import triton
import triton.language as tl

N = 4096
DST_ROWS = 1024
MAX_A = 12
A_CAPTURE = 6
BLOCK = 1024
NUM_LAYERS = 4
SENTINEL = -777.0


# ---------------------------------------------------------------- #
# 全局单例: 进程内唯一的 a_gpu(EP16 每 rank 一个进程, 各自独立)
# ---------------------------------------------------------------- #
@lru_cache(maxsize=None)
def get_a_gpu() -> torch.Tensor:
    return torch.zeros(1, dtype=torch.int32, device="cuda")


def set_num_rows(A: int):
    """runner 层每 step 调一次。绝不能在被 capture 的 forward 内部调用,
    否则 fill_ 的标量会被冻结进 graph。"""
    get_a_gpu().fill_(A)


# ---------------------------------------------------------------- #
# triton kernel: A 从显存读, grid 固定 MAX_A
# ---------------------------------------------------------------- #
@triton.jit
def dynamic_rows_copy_kernel(
    src_ptr, dst_ptr, a_ptr,
    src_stride0, dst_stride0,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    a = tl.load(a_ptr)  # replay 时唯一的动态信息来源

    if pid_row < a:
        offs = pid_col * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        v = tl.load(src_ptr + pid_row * src_stride0 + offs, mask=mask)
        tl.store(dst_ptr + pid_row * dst_stride0 + offs, v, mask=mask)


# ---------------------------------------------------------------- #
# 模拟 Transformer 里的 MLP 层: 不接收 a_gpu 参数, 内部自取
# ---------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_src = torch.zeros(MAX_A, N, device="cuda", dtype=torch.float32)
        self.dst = torch.full((DST_ROWS, N), SENTINEL,
                              device="cuda", dtype=torch.float32)

    def forward(self):
        a_gpu = get_a_gpu()  # ← 关键: 零侵入, 不经过 model forward 传参
        grid = (MAX_A, triton.cdiv(N, BLOCK))
        dynamic_rows_copy_kernel[grid](
            self.static_src, self.dst, a_gpu,
            self.static_src.stride(0), self.dst.stride(0),
            N=N, BLOCK=BLOCK,
        )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(MLP() for _ in range(NUM_LAYERS))

    def forward(self):
        for layer in self.layers:
            layer()


# ---------------------------------------------------------------- #
# runner 侧逻辑
# ---------------------------------------------------------------- #
def prepare_step(model: Model, A: int):
    """每 step 在 replay/forward 之前做: 填动态 src + 更新全局 a_gpu(一次)"""
    refs = [torch.randn(A, N, device="cuda", dtype=torch.float32)
            for _ in range(NUM_LAYERS)]
    for layer, ref in zip(model.layers, refs):
        layer.dst.fill_(SENTINEL)
        if A > 0:
            layer.static_src[:A].copy_(ref)
    set_num_rows(A)  # 全模型只这一处, 不在任何 layer 里
    return refs


def verify(model: Model, refs, A: int, tag: str):
    for i, (layer, ref) in enumerate(zip(model.layers, refs)):
        if A > 0:
            assert torch.equal(layer.dst[:A], ref), \
                f"[{tag} A={A}] layer{i} 前 {A} 行不一致"
        assert torch.all(layer.dst[A:] == SENTINEL), \
            f"[{tag} A={A}] layer{i} 第 {A} 行后被越界写"
    print(f"[PASS] {tag:>6s} A={A:>2d} x {NUM_LAYERS} layers")


def main():
    assert torch.cuda.is_available(), "需要 GPU 环境"
    torch.manual_seed(0)

    model = Model()
    get_a_gpu()  # 确保单例在 capture 前分配好

    # ---- eager 模式: 同一套代码路径, 不走 graph ----
    for A in [5, 0, 12]:
        refs = prepare_step(model, A)
        model()
        torch.cuda.synchronize()
        verify(model, refs, A, "eager")

    # ---- warmup(side stream, 触发 triton JIT)----
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        set_num_rows(A_CAPTURE)
        for _ in range(3):
            model()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # ---- capture: 整个 model forward, 内部 NUM_LAYERS 个 kernel ----
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model()
    print(f"capture 完成 (A_capture={A_CAPTURE}, {NUM_LAYERS} layers 共享 lru_cache 单例)")

    # ---- replay: 小于 / 等于 / 大于 capture 的 A + 0 边界 ----
    for A in [3, A_CAPTURE, 12, 0]:
        refs = prepare_step(model, A)   # set_num_rows 在 graph 外
        graph.replay()
        torch.cuda.synchronize()
        verify(model, refs, A, "replay")

    # ---- 健壮性: 确认单例地址从未变过 ----
    assert get_a_gpu().data_ptr() == get_a_gpu().data_ptr()
    print("\n全部用例通过")


if __name__ == "__main__":
    main()