"""
case04b: LLM decode 的 CUDA Graph 对比 —— 调成真正 launch-bound，让 Graph 收益明显。

case04 为什么不明显(看那次的 profiler 表)：
  aten::mm 的 "CUDA time avg" = 97.9us，远大于 launch 开销(~5us)。
  decode 的 GEMM 退化成 gemv，是【显存带宽受限】(每个 kernel 要读整块权重)，
  权重越大读得越久，GPU 真的在忙，launch 占比 <5% -> Graph 没东西可省。

判断 launch-bound 的硬指标：profiler 表里每个 kernel 的 "CUDA time avg"
  >> ~5us  -> compute/memory-bound，Graph 无用
  <= ~5us  -> launch-bound，Graph 有效

本 case 的做法：把每层维度调【小】(权重小、gemv 快到几 us) + 层数堆【多】(kernel 数量大)，
  让单 kernel GPU 时间掉到和 launch 开销同量级 -> eager 被 launch 拖住，replay 一次 launch 填平。
  想复现"不明显"，把 BIG=True 切回大模型即可对照。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from loguru import logger
from torch.profiler import profile, ProfilerActivity, record_function

DEV = "cuda"
DTYPE = torch.bfloat16

# ---- 两套配置：LAUNCH-BOUND(默认，Graph 明显) vs BIG(对照，Graph 无用) ----
BIG = False
if BIG:
    CFG = dict(d=2048, n_heads=16, head_dim=128, inter=5632, n_layers=16, max_seq=512)
else:
    # 小维度 -> 单 gemv 只要几 us；32 层 -> 每步几百个微型 kernel -> launch-bound
    CFG = dict(d=256, n_heads=4, head_dim=64, inter=512, n_layers=32, max_seq=128)


def warmup_on_side_stream(fn: Callable, iters: int = 3):
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(iters):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        n = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return n * self.w


class DecoderLayer(nn.Module):
    def __init__(self, d, n_heads, head_dim, inter, max_seq):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.attn_norm = RMSNorm(d)
        self.q = nn.Linear(d, n_heads * head_dim, bias=False)
        self.k = nn.Linear(d, n_heads * head_dim, bias=False)
        self.v = nn.Linear(d, n_heads * head_dim, bias=False)
        self.o = nn.Linear(n_heads * head_dim, d, bias=False)
        self.mlp_norm = RMSNorm(d)
        self.gate = nn.Linear(d, inter, bias=False)
        self.up = nn.Linear(d, inter, bias=False)
        self.down = nn.Linear(inter, d, bias=False)
        self.register_buffer("k_cache", torch.zeros(1, n_heads, max_seq, head_dim))
        self.register_buffer("v_cache", torch.zeros(1, n_heads, max_seq, head_dim))

    def forward(self, x, pos: int):
        h = self.attn_norm(x)
        q = self.q(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(1, 1, self.n_heads, self.head_dim).transpose(1, 2)
        self.k_cache[:, :, pos:pos + 1, :].copy_(k)
        self.v_cache[:, :, pos:pos + 1, :].copy_(v)
        attn = F.scaled_dot_product_attention(
            q, self.k_cache[:, :, :pos + 1, :], self.v_cache[:, :, :pos + 1, :])
        attn = attn.transpose(1, 2).reshape(1, 1, self.n_heads * self.head_dim)
        x = x + self.o(attn)
        h2 = self.mlp_norm(x)
        x = x + self.down(F.silu(self.gate(h2)) * self.up(h2))
        return x


class Decoder(nn.Module):
    def __init__(self, d, n_heads, head_dim, inter, n_layers, max_seq):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d, n_heads, head_dim, inter, max_seq) for _ in range(n_layers)])
        self.norm = RMSNorm(d)

    def forward(self, x, pos: int):
        for layer in self.layers:
            x = layer(x, pos)
        return self.norm(x)


def run(decode_steps: int = 50, pos: int = 64, trace_path: str = "trace_llm_decode.json"):
    logger.critical(f"decode profile | BIG={BIG} | {CFG['n_layers']} layers d={CFG['d']} inter={CFG['inter']}")
    pos = min(pos, CFG["max_seq"] - 1)

    model = Decoder(**CFG).to(DEV).to(DTYPE).eval()
    static_in = torch.randn(1, 1, CFG["d"], device=DEV, dtype=DTYPE)

    def decode_one():
        with torch.no_grad():
            return model(static_in, pos)

    warmup_on_side_stream(decode_one, iters=3)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = decode_one()
    with torch.no_grad():
        for _ in range(3):
            _ = model(static_in, pos)
    g.replay(); torch.cuda.synchronize()

    def time_region(fn_call) -> float:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        for _ in range(decode_steps):
            fn_call()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e)

    eager_ms = time_region(lambda: decode_one())
    replay_ms = time_region(lambda: g.replay())
    logger.debug(f"eager : {eager_ms/decode_steps:.3f} ms/token")
    logger.debug(f"replay: {replay_ms/decode_steps:.3f} ms/token")
    logger.debug(f"speedup: {eager_ms/replay_ms:.2f}x")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True, with_modules=True) as prof:
        with record_function("eager_decode"):
            with torch.no_grad():
                for _ in range(decode_steps):
                    _ = model(static_in, pos)
            torch.cuda.synchronize()
        with record_function("cudagraph_decode"):
            for _ in range(decode_steps):
                g.replay()
            torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    logger.debug(f"perfetto trace: {trace_path}")

    # ---- 诊断：单个 mm kernel 的 GPU 时间 vs launch 开销，判断处在哪个区间 ----
    ka = prof.key_averages()
    mm = next((e for e in ka if e.key == "aten::mm"), None)
    if mm is not None:
        us = mm.self_device_time_total / max(mm.count, 1)
        verdict = "launch-bound (Graph 有效)" if us <= 8 else "compute/memory-bound (Graph 无用)"
        logger.debug(f"诊断: 单 aten::mm GPU 时间 ~{us:.2f}us, launch 开销~5us -> {verdict}")
    logger.debug("\n" + ka.table(sort_by="cuda_time_total", row_limit=10))

    ref = decode_one(); torch.cuda.synchronize()
    logger.debug(f"max diff: {(static_out - ref).abs().max().item():.3e}")


if __name__ == "__main__":
    run()