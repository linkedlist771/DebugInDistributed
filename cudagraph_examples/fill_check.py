"""
最小可跑示例:CUDA graph 里用 Triton kernel 拷贝动态行数。

核心套路(NVIDIA "Dynamic Scalars" 标准解法):
  1. num_tokens 用一个持久的 device tensor 存,capture 前分配一次,全程复用(pointer 稳定)。
  2. kernel 里用 tl.load(sz_ptr) 在 replay 时才读当前值 —— 而不是把 int 当 launch 参数烧进 graph。
  3. fill_(host_int) 在 capture 区域【之外】更新标量,async、无 D2H sync。
  4. grid 用静态上界 max_size,真实拷贝行数靠 mask = offs < sz 动态裁剪。

跑法:python memcpy_triton_cudagraph_min.py   (需要 GPU + triton)
注:代码按 torch.cuda 写;MACA 上若 torch 把设备映射到 cuda 命名空间可直接跑,
    否则把 torch.cuda.* / 'cuda' 换成你们 MACA 对应的接口即可。
"""

import torch
import triton
import triton.language as tl
from math import prod
from loguru import logger

from functools import lru_cache


@lru_cache
def get_num_tokens_gpu() -> torch.Tensor:
    num_tokens_gpu = 

@triton.jit
def _memcpy_kernel(dst_ptr, src_ptr, sz_ptr, chunk_size, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(axis=0).to(tl.int64)
    sz   = tl.load(sz_ptr).to(tl.int64) * chunk_size      # 行数 -> 元素数,replay 时在 device 上读
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < sz
    data = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, data, mask=mask)


def memcpy_triton(dst, src, num_tokens_gpu):
    """num_tokens_gpu: GPU int32 tensor [1],当前有效行数(不是元素数)。"""
    assert dst.is_contiguous() and src.is_contiguous(), "must be contiguous"
    assert num_tokens_gpu.is_cuda, "num_tokens must be GPU tensor"
    chunk_size = prod(src.shape[1:]) if src.dim() > 1 else 1
    BLOCK_SIZE = 8192
    max_size = min(src.numel(), dst.numel())              # 静态上界,capture 时固定
    grid = (triton.cdiv(max_size, BLOCK_SIZE),)
    _memcpy_kernel[grid](dst, src, num_tokens_gpu, chunk_size, BLOCK_SIZE)


def main():
    device = "cuda"
    MAX_ROWS, DIM = 1024, 128

    # ---- 持久 buffer:capture 前分配一次,全程复用 ----
    # src 第 i 行的元素全等于 i,方便验证拷了哪些行
    src = (torch.arange(MAX_ROWS, device=device, dtype=torch.float32)
           .unsqueeze(1).expand(MAX_ROWS, DIM).contiguous())
    dst = torch.empty(MAX_ROWS, DIM, device=device, dtype=torch.float32)
    num_tokens_gpu = torch.zeros(1, dtype=torch.int32, device=device)   # << 关键:持久标量

    # ---- warmup:必须在 side stream 上,顺便触发 triton 编译(capture 时不能再编译)----
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            num_tokens_gpu.fill_(MAX_ROWS)
            dst.zero_()
            memcpy_triton(dst, src, num_tokens_gpu)
    torch.cuda.current_stream().wait_stream(s)

    # ---- capture:graph 里只录稳定指针上的 kernel;fill_ 不在这里 ----
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        dst.zero_()                       # 每次 replay 先清零(静态 memset,大小固定)
        memcpy_triton(dst, src, num_tokens_gpu)

    # ---- replay:fill_ 在 capture 区域之外更新标量,然后 replay ----
    logger.debug(f"{'n':>6} | copied_rows_ok | rest_is_zero")
    logger.debug("-" * 40)
    for n in [8, 100, 777, 1024]:
        num_tokens_gpu.fill_(n)           # host int -> device,async,无 sync
        g.replay()                        # kernel 里 tl.load(sz_ptr) 读到的就是新的 n
        torch.cuda.synchronize()

        ok_copied = torch.equal(dst[:n], src[:n])              # 前 n 行 == src
        ok_rest   = (torch.count_nonzero(dst[n:]).item() == 0)  # 其余行保持 0
        logger.debug(f"{n:>6} | {str(ok_copied):>14} | {str(ok_rest):>12}")

    logger.debug("\n预期:四行全部 True True —— 同一张 graph,replay 间靠 fill_ 改变拷贝行数。")


if __name__ == "__main__":
    main()