"""
case05: 从 DeepSeek-V4 提取 MoE 核心，改成 graph-safe 版本，演示 CUDA Graph replay 效果。

为什么原版 DeepseekV4Experts.forward 不能直接捕获：
    hit = (...).nonzero()          # 数据相关的动态 shape + host 同步
    for expert_idx in hit:         # Python 循环次数由运行时数据决定
        token_idx = torch.where()  # 又是动态 shape
        final.index_add_(...)
  -> 动态 shape / host 同步 / 数据相关控制流，三样都是 CUDA Graph 的禁忌。

graph-safe 改写的两条原则(也是 MegaMaskedGroupedGEMM 的原则)：
  1. 固定 shape、无 host 同步：用静态掩码 + bmm 计算，topk/scatter 都是定 shape kernel。
  2. 输出 buffer 必须 zero 初始化(不是 torch.empty!)：
     masked/grouped GEMM 对未命中的 expert 行【不写入】，empty 的垃圾值会留下来 ->
     NaN / 脏输出。这就是你之前那个 _run_mctlass_mega_routed 的真实 bug。

decode 阶段(batch=1 seq=1)下，MoE 每层有大量小 kernel(路由 + 每专家的 bmm + 共享专家 +
多个 RMSNorm + HC 残差) -> launch-bound -> CUDA Graph replay 收益明显。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from loguru import logger
from torch.profiler import profile, ProfilerActivity, record_function

DEV = "cuda"
DTYPE = torch.bfloat16


def warmup_on_side_stream(fn: Callable, iters: int = 3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(iters):
            fn()
    torch.cuda.current_stream().wait_stream(s)


# ---------- 从 DeepSeek-V4 原样提取 ----------
class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dt)


class DeepseekV4TopKRouter(nn.Module):
    """提取自原版：sigmoid 打分 + e_score_correction_bias + routed_scaling_factor。"""
    def __init__(self, hidden, num_experts, top_k, scaling=2.5):
        super().__init__()
        self.top_k, self.num_experts, self.hidden_dim = top_k, num_experts, hidden
        self.weight = nn.Parameter(torch.empty(num_experts, hidden))
        nn.init.normal_(self.weight, std=0.02)
        self.routed_scaling_factor = scaling
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts))

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = logits.sigmoid()
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * self.routed_scaling_factor, indices


class DeepseekV4MLP(nn.Module):
    """共享专家(原版结构：clamp 后 SwiGLU)。"""
    def __init__(self, hidden, inter, limit=7.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.limit = limit

    def forward(self, x):
        gate = self.gate_proj(x).clamp(max=self.limit)
        up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


# ---------- 核心：graph-safe 的 Experts(替换原版的数据相关循环) ----------
class GraphSafeExperts(nn.Module):
    """权重布局沿用原版 3D 张量(gate_up_proj[E,2I,H], down_proj[E,H,I])，
    但 forward 改成【静态 shape、无 host 同步】，可被 CUDA Graph 捕获。

    做法：所有 token 过所有 expert(dense bmm)，再用路由权重(未选中=0)做掩码加权。
    代价是算了全部 expert(O(E))，但 shape 全静态、无 .nonzero/无 python 循环。
    生产里 MegaMaskedGroupedGEMM 用 masked grouped GEMM 只算选中的 expert，
    原则相同：固定 shape + 输出 buffer 必须 zero 初始化。
    """
    def __init__(self, num_experts, hidden, inter, limit=7.0):
        super().__init__()
        self.num_experts, self.hidden, self.inter, self.limit = num_experts, hidden, inter, limit
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, inter))
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, flat, top_k_index, top_k_weights):
        # flat: [T, H]；top_k_index/weights: [T, k]
        T, H = flat.shape
        E = self.num_experts

        # 路由权重展开成稠密 [T, E]，未选中的位置是 0 —— 这就是 graph-safe 的“掩码”
        # 注意 zeros 而非 empty：未命中的 expert 不参与，靠这里的 0 来屏蔽
        routing_full = torch.zeros(T, E, device=flat.device, dtype=flat.dtype)
        routing_full.scatter_(1, top_k_index, top_k_weights.to(flat.dtype))  # 静态 shape scatter

        # 所有 token 过所有 expert：固定 shape，无控制流
        x = flat.unsqueeze(0).expand(E, T, H).contiguous()          # [E, T, H]
        gate_up = torch.bmm(x, self.gate_up_proj.transpose(1, 2))    # [E, T, 2I]
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        act = F.silu(gate) * up                                       # [E, T, I]
        out = torch.bmm(act, self.down_proj.transpose(1, 2))         # [E, T, H]

        # 用路由权重(未选中=0)加权求和回 [T, H]
        w = routing_full.transpose(0, 1).unsqueeze(-1)               # [E, T, 1]
        return (out * w).sum(0)                                       # [T, H]


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, hidden, inter, num_experts, top_k):
        super().__init__()
        self.gate = DeepseekV4TopKRouter(hidden, num_experts, top_k)
        self.experts = GraphSafeExperts(num_experts, hidden, inter)
        self.shared_experts = DeepseekV4MLP(hidden, inter)

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, H)
        weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(B, S, H)
        return routed + self.shared_experts(residual)


# ---------- 轻量注意力(共享-KV MQA + 静态 KV cache) ----------
class Attention(nn.Module):
    """简化版：保留 DeepSeek 的单 KV head(MQA) + 静态 KV cache(地址固定，graph 安全)。
    省略了 CSA/HCA 压缩器和 partial-RoPE —— 压缩器输出长度可变，本身不可捕获，
    与本 demo(展示 replay)无关，故略去，把重点放在可捕获的 MoE 上。"""
    def __init__(self, hidden, n_heads, head_dim, max_seq):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.kv_proj = nn.Linear(hidden, head_dim, bias=False)  # 单 KV head
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.scale = head_dim ** -0.5
        self.register_buffer("k_cache", torch.zeros(1, 1, max_seq, head_dim))
        self.register_buffer("v_cache", torch.zeros(1, 1, max_seq, head_dim))

    def forward(self, x, pos: int):
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)
        self.k_cache[:, :, pos:pos + 1, :].copy_(kv)
        self.v_cache[:, :, pos:pos + 1, :].copy_(kv)
        k = self.k_cache[:, :, :pos + 1, :].expand(B, self.n_heads, pos + 1, self.head_dim)
        v = self.v_cache[:, :, :pos + 1, :].expand(B, self.n_heads, pos + 1, self.head_dim)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(attn)


class DecoderLayer(nn.Module):
    def __init__(self, hidden, inter, n_heads, head_dim, num_experts, top_k, max_seq):
        super().__init__()
        self.input_layernorm = DeepseekV4RMSNorm(hidden)
        self.self_attn = Attention(hidden, n_heads, head_dim, max_seq)
        self.post_attention_layernorm = DeepseekV4RMSNorm(hidden)
        self.mlp = DeepseekV4SparseMoeBlock(hidden, inter, num_experts, top_k)

    def forward(self, x, pos: int):
        x = x + self.self_attn(self.input_layernorm(x), pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MoEDecoder(nn.Module):
    def __init__(self, hidden=512, inter=256, n_heads=8, head_dim=64,
                 num_experts=16, top_k=4, n_layers=8, max_seq=128):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(hidden, inter, n_heads, head_dim, num_experts, top_k, max_seq)
            for _ in range(n_layers)
        ])
        self.norm = DeepseekV4RMSNorm(hidden)

    def forward(self, x, pos: int):
        for layer in self.layers:
            x = layer(x, pos)
        return self.norm(x)


def run(decode_steps=50, pos=64, trace_path="trace_deepseek_moe.json"):
    cfg = dict(hidden=512, inter=256, n_heads=8, head_dim=64,
               num_experts=16, top_k=4, n_layers=8, max_seq=128)
    pos = min(pos, cfg["max_seq"] - 1)
    logger.critical(f"DeepSeek-V4 MoE decode | {cfg['n_layers']} layers, {cfg['num_experts']} experts, top{cfg['top_k']}")

    model = MoEDecoder(**cfg).to(DEV).to(DTYPE).eval()
    static_in = torch.randn(1, 1, cfg["hidden"], device=DEV, dtype=DTYPE)

    def decode_one():
        with torch.no_grad():
            return model(static_in, pos)

    # 1) 侧流预热
    warmup_on_side_stream(decode_one, iters=3)

    # 2) 捕获整层 forward
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = decode_one()

    # 3) eager 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(static_in, pos)
    g.replay(); torch.cuda.synchronize()

    # CUDA event 量化(裸跑)
    def time_region(fn_call):
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

    # 带 profiler 导 trace
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

    ka = prof.key_averages()
    bmm = next((x for x in ka if x.key == "aten::bmm"), None)
    if bmm is not None:
        us = bmm.self_device_time_total / max(bmm.count, 1)
        verdict = "launch-bound (Graph 有效)" if us <= 8 else "compute-bound (Graph 弱)"
        logger.debug(f"诊断: 单 aten::bmm GPU 时间 ~{us:.2f}us, launch~5us -> {verdict}")
    logger.debug("\n" + ka.table(sort_by="cuda_time_total", row_limit=12))

    ref = decode_one(); torch.cuda.synchronize()
    logger.debug(f"replay vs eager 最大误差: {(static_out - ref).abs().max().item():.3e}")


if __name__ == "__main__":
    run()