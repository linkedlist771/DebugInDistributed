"""
================================================================================
 mega_moe_dist.py —— 对外 API 对齐官方 deep_gemm 的 Mega MoE 学习版(Gloo 多进程)
================================================================================

对外暴露的调用形式和官方一模一样,方便你照着学:

    import mega_moe                                              # 本文件即此"库"
    buffer = mega_moe.get_symm_buffer_for_mega_moe(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden)
    tl1, tl2 = mega_moe.transform_weights_for_mega_moe(l1_weights, l2_weights)
    buffer.x[:num_tokens].copy_(x_fp8)                           # 调用前把输入拷进 slot
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)
    y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=dev)
    mega_moe.fp8_fp4_mega_moe(y, tl1, tl2, buffer)              # 一个融合调用

每个进程 = 一个 rank,持有自己的 SymmBuffer,fp8_fp4_mega_moe 是 *per-rank* 调用
(这正是官方的形态:所以才需要 rank 间的 barrier)。

--------------------------------------------------------------------------------
 对照官方源码
--------------------------------------------------------------------------------
 deep_gemm/mega/__init__.py SymmBuffer / get_symm_buffer_for_mega_moe  →  本文件同名
 deep_gemm/mega/__init__.py transform_weights_for_mega_moe             →  本文件同名(占位)
 csrc/apis/mega.hpp fp8_fp4_mega_moe                                   →  本文件 fp8_fp4_mega_moe
 layout/mega_moe.cuh Workspace(counters / TokenSrcMetadata)            →  dispatch 里的 counts/meta
 comm/barrier.cuh nvlink_barrier                                       →  dist.barrier()(Gloo)
 layout/sym_buffer.cuh SymBuffer.map(ptr, dst_rank)(单边写)            →  见 LIMITATIONS:Gloo 没有

--------------------------------------------------------------------------------
 老实交代:Gloo vs 真 Mega MoE
--------------------------------------------------------------------------------
 真 Mega MoE 的传输是 GPU 发起的 *单边* NVLink 写(sym_buffer.map(dst) 直写对端 slot),
 且 dispatch warp 与 MMA warp 在同一 kernel 内 *并发*(warp 级 overlap)。
 Gloo 只有 *双边* 集合通信、没有 RMA、更没有 warp overlap。所以本文件:
   - dispatch/combine 的传输退化成 all_to_all(本质是 DeepEP 那种两步 dispatch);
   - barrier 用 dist.barrier();
   - 整个 kernel 顺序执行,无重叠。
 但 *API 形态 + 命名 slot buffer + 单函数融合 + contiguous pool + 融合 epilogue* 全部保留 —— 
 这些才是你照着官方学时真正要看的结构。性能形态(单边/overlap)需要 NVSHMEM/symm-mem + CUDA。

 设备:计算(GEMM/SwiGLU/量化)在 buffer 所在 device(你的 CUDA 卡);
 通信走 Gloo,统一在 CPU 上中转(Gloo 对 CUDA / float8 的 collective 支持不全,
 故把 FP8 bitcast 成 uint8、其余搬到 CPU 再 all_to_all,稳妥可跑)。
================================================================================
"""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ───────────────────────── 量化配置(与官方 recipe=(1,1,32) 对齐)─────────────────────────
SF_BLOCK = 32
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
ACT_CLAMP_DEFAULT = float("inf")


# ============================================================================
#  量化 helper(供 *调用方* 在 .copy_ 进 buffer 之前使用,对应官方的 x_fp8 / x_sf)
# ============================================================================
def quantize_fp8_ue8m0(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """x:[m,k] → (fp8 e4m3, scale:[m, k//SF_BLOCK] fp32,2 的幂 UE8M0)。"""
    m, k = x.shape
    assert k % SF_BLOCK == 0, f"k={k} 必须能被 SF_BLOCK={SF_BLOCK} 整除"
    xb = x.reshape(m, k // SF_BLOCK, SF_BLOCK).float()
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    xq = (xb / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return xq.reshape(m, k), scale.reshape(m, k // SF_BLOCK)


def _dequant_fp8(xq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = xq.shape
    xf = xq.float().reshape(m, k // SF_BLOCK, SF_BLOCK)
    return (xf * scale.reshape(m, k // SF_BLOCK, 1)).reshape(m, k)


# ============================================================================
#  SymmBuffer + get_symm_buffer_for_mega_moe(对齐官方)
# ============================================================================
class SymmBuffer:
    """一块常驻 buffer,切出命名 slot —— 对应官方 SymmBuffer。
       调用方往 .x/.x_sf/.topk_idx/.topk_weights 里 copy_;kernel 内部用
       l1/l2 池 slot 和 combine slot 做中间落地。"""

    def __init__(self, group: Optional["dist.ProcessGroup"],
                 num_experts: int, num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int, device: torch.device):
        self.group = group
        self.world_size = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        self.num_experts = num_experts
        self.num_max_tokens = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.experts_per_rank = num_experts // self.world_size
        self.device = device

        # 本地 expert token pool 容量(worst case,对应 layout/mega_moe.cuh get_num_max_pool_tokens)
        recv_max = self.world_size * num_max_tokens_per_rank
        per_tok = min(num_topk, self.experts_per_rank)
        self.pool_cap = recv_max * per_tok + self.experts_per_rank * 64

        H, I = hidden, intermediate_hidden
        dev = device

        # ── 输入 slot(调用方 copy_ 进来,落地即 FP8,对应 use_fp8_dispatch)──
        self.x            = torch.zeros(num_max_tokens_per_rank, H, dtype=FP8_DTYPE, device=dev)
        self.x_sf         = torch.zeros(num_max_tokens_per_rank, H // SF_BLOCK, device=dev)
        self.topk_idx     = torch.zeros(num_max_tokens_per_rank, num_topk, dtype=torch.long, device=dev)
        self.topk_weights = torch.zeros(num_max_tokens_per_rank, num_topk, device=dev)
        # 非官方 slot:bf16 影子输入,仅用于 quantize=False 的"流水线塌缩"精确校验
        self.x_bf16       = torch.zeros(num_max_tokens_per_rank, H, device=dev)

        # ── L1 池 slot(dispatch 落地处)──
        self.l1_acts    = torch.zeros(self.pool_cap, H, dtype=FP8_DTYPE, device=dev)
        self.l1_acts_sf = torch.zeros(self.pool_cap, H // SF_BLOCK, device=dev)
        self.l1_topk_wt = torch.zeros(self.pool_cap, device=dev)

        # ── L2 池 slot(L1 epilogue 输出 = GEMM2 输入,住同一 buffer)──
        self.l2_acts    = torch.zeros(self.pool_cap, I, dtype=FP8_DTYPE, device=dev)
        self.l2_acts_sf = torch.zeros(self.pool_cap, I // SF_BLOCK, device=dev)

        # ── combine slot(GEMM2 结果暂存,按 topk 归约)──
        self.combine = torch.zeros(num_topk, num_max_tokens_per_rank, H, device=dev)


def get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank,
                                 num_topk, hidden, intermediate_hidden,
                                 device: Optional[torch.device] = None) -> SymmBuffer:
    assert num_experts % (dist.get_world_size(group) if group else 1) == 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SymmBuffer(group, num_experts, num_max_tokens_per_rank,
                      num_topk, hidden, intermediate_hidden, device)


def transform_weights_for_mega_moe(l1_weights: torch.Tensor, l2_weights: torch.Tensor):
    """对应官方 transform_weights_for_mega_moe:
       官方把 L1 的 gate/up 按粒度 8 交错、SF 为 UTCCP 转置(都是硬件 layout 的事)。
       本学习版跑普通 matmul 不需要这些,做恒等占位,保留注释说明其存在意义。
       约定 l1:[experts_per_rank, 2I, H](gate||up),l2:[experts_per_rank, H, I]。"""
    return l1_weights, l2_weights


# ============================================================================
#  Gloo all_to_all helper(统一 CPU 中转;FP8 bitcast 成 uint8)
# ============================================================================
def _a2a_splits(group, x: torch.Tensor, in_splits, out_splits) -> torch.Tensor:
    dev = x.device
    is_fp8 = (x.dtype == FP8_DTYPE)
    xc = (x.view(torch.uint8) if is_fp8 else x).cpu().contiguous()
    out = torch.empty((sum(out_splits), *xc.shape[1:]), dtype=xc.dtype)
    dist.all_to_all_single(out, xc,
                           output_split_sizes=list(out_splits),
                           input_split_sizes=list(in_splits),
                           group=group)
    out = out.to(dev)
    return out.view(FP8_DTYPE) if is_fp8 else out


def _exchange_counts(group, send_counts: torch.Tensor) -> torch.Tensor:
    """把 send_counts 换成 recv_counts(经典两步 dispatch 的第一步)。
       对应官方 dispatch 用 NVLink 把 per-expert count 写给对端,这里用一次 all_to_all。"""
    sc = send_counts.cpu().contiguous()
    rc = torch.empty_like(sc)
    dist.all_to_all_single(rc, sc, group=group)
    return rc.to(send_counts.device)


# ============================================================================
#  fp8_fp4_mega_moe —— 单个融合调用(per-rank),对应 sm100_fp8_fp4_mega_moe.cuh
# ============================================================================
@torch.no_grad()
def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: torch.Tensor, l2_weights: torch.Tensor,
                     buffer: SymmBuffer,
                     activation_clamp: float = ACT_CLAMP_DEFAULT,
                     quantize: bool = True):
    """y:[num_tokens, hidden] bf16 输出(num_tokens 从 y 推断)。
       l1:[epr,2I,H],l2:[epr,H,I] 为 *本 rank 的本地专家* 权重。
       整条 dispatch→GEMM1→SwiGLU+gate+量化→GEMM2→combine 全在此函数内。"""
    group = buffer.group
    ws = buffer.world_size
    rank = buffer.rank
    epr = buffer.experts_per_rank
    H, I = buffer.hidden, buffer.intermediate_hidden
    TOPK = buffer.num_topk
    dev = buffer.device
    n = y.size(0)

    # 从 buffer slot 取输入(调用方已 copy_ 进来)
    x_fp8 = buffer.x[:n]                 # [n,H] fp8
    x_sf  = buffer.x_sf[:n]              # [n,H//SF]
    topk_idx = buffer.topk_idx[:n]       # [n,TOPK]
    topk_wt  = buffer.topk_weights[:n]   # [n,TOPK]

    # ───────────────── 阶段 A:dispatch(all_to_all 到 owner rank)─────────────────
    # 展开成 P = n*TOPK 个 (token, topk) pair,按目标 rank 分组
    P = n * TOPK
    flat_e = topk_idx.reshape(-1)                       # [P] 全局专家
    flat_w = topk_wt.reshape(-1)                        # [P] gate 权重
    pair_tok = torch.arange(n, device=dev).repeat_interleave(TOPK)  # [P] 源 token
    dst = (flat_e // epr).long()                        # [P] 目标 rank
    local_e = (flat_e % epr).long()                     # [P] 在目标 rank 的本地专家

    send_perm = torch.argsort(dst, stable=True)         # 按 dst 分组
    send_counts = torch.bincount(dst, minlength=ws)     # 每个 dst 发多少
    recv_counts = _exchange_counts(group, send_counts)  # ← 对端要收多少
    sc, rc = send_counts.tolist(), recv_counts.tolist()
    recv_total = sum(rc)

    # 按 send_perm 排好的 payload(激活精度由 quantize 决定:FP8 真实路径 / bf16 校验路径)
    if quantize:
        s_act = x_fp8.repeat_interleave(TOPK, dim=0)[send_perm]    # [P,H] fp8
        s_sf  = x_sf.repeat_interleave(TOPK, dim=0)[send_perm]     # [P,H//SF]
    else:
        s_act = buffer.x_bf16[:n].repeat_interleave(TOPK, dim=0)[send_perm]  # [P,H] bf16 影子
    s_wt  = flat_w[send_perm].unsqueeze(1)                         # [P,1]
    s_le  = local_e[send_perm].unsqueeze(1).int()                 # [P,1] 本地专家

    # all_to_all 数据(FP8 走 uint8 bitcast;bf16 影子走 float)
    r_act = _a2a_splits(group, s_act, sc, rc)                     # [recv_total,H]
    r_wt  = _a2a_splits(group, s_wt,  sc, rc).squeeze(1)          # [recv_total]
    r_le  = _a2a_splits(group, s_le,  sc, rc).squeeze(1).long()   # [recv_total]
    if quantize:
        r_sf = _a2a_splits(group, s_sf, sc, rc)                  # [recv_total,H//SF]
    if group is not None:
        dist.barrier(group)   # 对应 nvlink_barrier:保证 dispatch 落地对所有 rank 可见

    if recv_total == 0:
        y.zero_()
        return

    # ── 落进 buffer 的 L1 池 slot(住常驻 buffer,非每次新建)──
    buffer.l1_topk_wt[:recv_total] = r_wt
    if quantize:
        buffer.l1_acts[:recv_total]    = r_act
        buffer.l1_acts_sf[:recv_total] = r_sf

    # ── 按本地专家排成 contiguous 段(grouped GEMM 输入形态)──
    ex_perm = torch.argsort(r_le, stable=True)
    inv_ex = torch.empty_like(ex_perm); inv_ex[ex_perm] = torch.arange(recv_total, device=dev)
    counts_e = torch.bincount(r_le, minlength=epr).tolist()
    seg = [0]
    for c in counts_e:
        seg.append(seg[-1] + c)

    gate_wt = buffer.l1_topk_wt[:recv_total][ex_perm]
    if quantize:
        a_deq = _dequant_fp8(buffer.l1_acts[:recv_total][ex_perm],
                             buffer.l1_acts_sf[:recv_total][ex_perm])
    else:
        a_deq = r_act[ex_perm].float()        # bf16 影子,直接用(应用了正确数值)

    # ───────────────── 阶段 B:GEMM1(grouped contiguous)─────────────────
    l1_out = torch.zeros(recv_total, 2 * I, device=dev)
    for le in range(epr):
        s, t = seg[le], seg[le + 1]
        if t > s:
            l1_out[s:t] = a_deq[s:t].to(torch.float32) @ l1_weights[le].T.float()

    # ───────────────── 阶段 C:融合 epilogue(SwiGLU + gate 加权 + clamp + 量化)──────────
    gate, up = l1_out[:, :I], l1_out[:, I:]
    if activation_clamp != float("inf"):
        gate = gate.clamp(-activation_clamp, activation_clamp)
        up   = up.clamp(-activation_clamp, activation_clamp)
    act = F.silu(gate) * up
    act = act * gate_wt.unsqueeze(-1)                 # ★ gate 加权前移到中间激活(免 combine 再乘)
    if quantize:
        l2q, l2sf = quantize_fp8_ue8m0(act)
        buffer.l2_acts[:recv_total][ex_perm] = l2q    # 写回 buffer slot
        buffer.l2_acts_sf[:recv_total][ex_perm] = l2sf
        a2_deq = _dequant_fp8(l2q, l2sf) # 在计算前都进行反量化
    else:
        a2_deq = act

    # ───────────────── 阶段 D:GEMM2(grouped contiguous)──────────────────
    l2_out = torch.zeros(recv_total, H, device=dev)
    for le in range(epr):
        s, t = seg[le], seg[le + 1]
        if t > s:
            l2_out[s:t] = a2_deq[s:t].to(torch.float32) @ l2_weights[le].T.float()

    # 撤销 expert 分段排序 → recv 顺序
    l2_recvorder = l2_out[inv_ex]

    # ───────────────── 阶段 E:combine(反向 all_to_all 写回原 rank)──────────
    # 真 kernel:按 TokenSrcMetadata 单边写回 src rank;Gloo 用反向 all_to_all(splits 互换)
    l2_back = _a2a_splits(group, l2_recvorder, rc, sc)    # [P,H] 回到 send 顺序(按 dst 分组)
    if group is not None:
        dist.barrier(group)

    # 撤销 send_perm → 原 (token,topk) pair 顺序,再按 token 归约(gate 已在 epilogue 乘过)
    inv_send = torch.empty_like(send_perm); inv_send[send_perm] = torch.arange(P, device=dev)
    l2_pairorder = l2_back[inv_send]                      # [P,H]
    y.zero_()
    y.index_add_(0, pair_tok, l2_pairorder.to(y.dtype))   # 沿 topk 累加 → [n,H]


# ============================================================================
#  参考实现 + 多进程测试(Gloo)
# ============================================================================
def _reference(x, topk_idx, topk_wt, l1_full, l2_full, epr, ws, H, I, TOPK, clamp):
    n = x.size(0)
    out = torch.zeros(n, H)
    xf = x.float()
    for tk in range(n):
        for j in range(TOPK):
            e = int(topk_idx[tk, j])
            w = l1_full[e].float() @ xf[tk]
            gate, up = w[:I], w[I:]
            if clamp != float("inf"):
                gate = gate.clamp(-clamp, clamp); up = up.clamp(-clamp, clamp)
            act = F.silu(gate) * up
            yk = l2_full[e].float() @ act
            out[tk] += topk_wt[tk, j] * yk
    return out


def _worker(rank, world_size, E, H, I, TOPK, n_tokens, quantize, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    group = dist.group.WORLD
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(0)          # 单卡:所有进程共用 device 0

    epr = E // world_size
    clamp = 7.0

    # 全局权重(各 rank 同种子 → 一致);kernel 只用本 rank 的本地切片
    g = torch.Generator().manual_seed(1234)
    l1_full = (torch.randn(E, 2 * I, H, generator=g) * 0.1)
    l2_full = (torch.randn(E, H, I, generator=g) * 0.1)
    l1_local = l1_full[rank * epr:(rank + 1) * epr].to(dev)
    l2_local = l2_full[rank * epr:(rank + 1) * epr].to(dev)

    # 路由
    Wg = torch.randn(E, H, generator=torch.Generator().manual_seed(7)) * 0.1
    x = torch.randn(n_tokens, H, generator=torch.Generator().manual_seed(100 + rank))
    tw, ti = torch.softmax(x @ Wg.T, dim=-1).topk(TOPK, dim=-1)
    ref = _reference(x, ti, tw, l1_full, l2_full, epr, world_size, H, I, TOPK, clamp)

    # ===== 官方形态的调用 =====
    max_tokens = n_tokens
    buffer = get_symm_buffer_for_mega_moe(group, E, max_tokens, TOPK, H, I, device=dev)
    tl1, tl2 = transform_weights_for_mega_moe(l1_local, l2_local)
    x_fp8, x_sf = quantize_fp8_ue8m0(x.to(dev))
    buffer.x[:n_tokens].copy_(x_fp8)
    buffer.x_sf[:n_tokens].copy_(x_sf)
    buffer.x_bf16[:n_tokens].copy_(x.to(dev))   # bf16 影子(仅 quantize=False 校验用)
    buffer.topk_idx[:n_tokens].copy_(ti.to(dev))
    buffer.topk_weights[:n_tokens].copy_(tw.to(dev))
    y = torch.empty((n_tokens, H), dtype=torch.bfloat16, device=dev)
    fp8_fp4_mega_moe(y, tl1, tl2, buffer, activation_clamp=clamp, quantize=quantize)

    yc = y.float().cpu()
    abs_max = (yc - ref).abs().max().item()
    rel = ((yc - ref).norm() / ref.norm().clamp_min(1e-9)).item()
    ret[rank] = (abs_max, rel)
    dist.barrier(group)
    if rank == 0:
        print(f"[ws={world_size} quantize={quantize}] device={dev.type}")
    dist.barrier(group)
    print(f"  rank {rank}: abs_max={abs_max:.3e}  rel_l2={rel:.2%}")
    dist.destroy_process_group()


def main(world_size=2, quantize=True):
    import torch.multiprocessing as mp
    E, H, I, TOPK, n_tokens = 8, 256, 512, 2, 6
    assert E % world_size == 0
    mgr = mp.Manager(); ret = mgr.dict()
    mp.spawn(_worker, args=(world_size, E, H, I, TOPK, n_tokens, quantize, ret),
             nprocs=world_size, join=True)
    abs_max = max(v[0] for v in ret.values())
    rel = max(v[1] for v in ret.values())
    if quantize:
        print(f"  => global rel_l2={rel:.2%}  {'OK(量化误差有界)' if rel < 0.10 else 'TOO LARGE!'}\n")
    else:
        # quantize=False:激活/GEMM 全 fp32,唯一损失是 y 的 bf16 输出取整(~0.4% eps)。
        # rel_l2 在 1% 以内即证明 dispatch/pool/grouped-GEMM/combine 的路由与归约完全正确。
        print(f"  => global rel_l2={rel:.2%}  {'OK(流水线塌缩正确,残差=bf16输出取整)' if rel < 0.01 else 'MISMATCH!'}\n")


if __name__ == "__main__":
    main(world_size=2, quantize=False)   # 先验证路由/归约正确(bf16)
    main(world_size=4, quantize=False)
    main(world_size=2, quantize=True)    # 再验证 FP8 路径(误差有界)
    main(world_size=4, quantize=True)


# ============================================================================
#  LIMITATIONS(对比真 Mega MoE)
# ============================================================================
#  搬不动:
#   - 单边 NVLink 写(sym_buffer.map(dst)):Gloo 无 RMA → 退化为 all_to_all(两步 dispatch)
#   - warp 级 overlap:dispatch 与 GEMM 顺序执行,无重叠
#   - 片上中间激活(TMEM 直传):l1/l2 仍过 buffer(至少不每步新建)
#   - tcgen05/TMA/UTCCP、mxf8f6f4/MXF4 mainloop、persistent scheduler:需 CUDA kernel
#   - FP4 权重:本版权重保持高精度做 matmul,仅激活走真 FP8 + UE8M0 scale
#  搬动了(照官方学的重点):
#   - 对外 API 形态(get_symm_buffer / copy_ slot / transform_weights / fp8_fp4_mega_moe)
#   - 命名 slot 的常驻对称 buffer
#   - 整条 MoE 塌缩进一个 per-rank 融合函数
#   - 融合 epilogue(SwiGLU + gate 加权前移 + clamp + FP8 量化一处做完)
#   - 按 expert 的 contiguous token pool
# ============================================================================