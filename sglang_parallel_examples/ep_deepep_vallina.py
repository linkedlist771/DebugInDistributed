import torch
import torch.distributed as dist
import torch.nn.functional as F
from common import cleanup, launch, rprint, setup

H, I, E = 16, 32, 8


def expert_weights(eid):
    g = torch.Generator().manual_seed(2000 + eid)
    return torch.randn(I, H, generator=g), torch.randn(H, I, generator=g)

def run_expert(x, eid):
    w_in, w_out = expert_weights(eid)
    return F.silu(x @ w_in.T) @ w_out.T


# ────────── 原理 1+2+3:把 Buffer 显式化、一次分配、handle 承载状态 ──────────
class DispatchHandle:
    """对应 DeepEP dispatch 返回的 handle:把 combine 需要的一切都存下来。"""
    __slots__ = ("perm", "send_counts", "recv_counts", "gate_sorted",
                 "expert_perm", "num_recv_per_expert", "n_recv")


class EPBuffer:
    def __init__(self, group, world_size, max_tokens, hidden, dtype):
        self.group, self.ws = group, world_size
        self.experts_per_rank = E // world_size
        # ★ 原理 1:一次性预分配,worst-case 接收 = 所有 rank 都发给我
        max_recv = world_size * max_tokens
        self._send_buf = torch.empty(max_tokens, hidden, dtype=dtype)
        self._recv_buf = torch.empty(max_recv,  hidden, dtype=dtype)
        self._eid_send = torch.empty(max_tokens, 1, dtype=torch.float32)
        self._eid_recv = torch.empty(max_recv,  1, dtype=torch.float32)
        self._cnt_recv = torch.empty(world_size, dtype=torch.float32)

    # ── 内部:一次变长 all_to_all,但写进常驻 buffer 的切片(不新建) ──
    def _a2a_into(self, buf, src, send_counts, recv_counts):
        n = int(recv_counts.sum())
        out = buf[:n]                                  # ★ 切片复用,非 torch.empty
        dist.all_to_all_single(
            out, src.contiguous(),
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=self.group)
        return out

    # ★ 原理 2:layout 计算独立成步(对应 get_dispatch_layout + a2a_counts)
    def get_dispatch_layout(self, expert_id):
        dest = expert_id // self.experts_per_rank
        send_counts = torch.bincount(dest, minlength=self.ws)
        recv_counts = self._cnt_recv                    # 复用 buffer
        dist.all_to_all_single(recv_counts, send_counts.float(), group=self.group)
        recv_counts = recv_counts.long()
        return dest, send_counts, recv_counts

    # ★ 原理 3+4:dispatch 搬数据 + 在接收端排成 contiguous 分组 layout
    def dispatch(self, x, expert_id, gate, my_first_expert):
        dest, send_counts, recv_counts = self.get_dispatch_layout(expert_id)

        perm = torch.argsort(dest)
        x_sorted   = x[perm]
        eid_sorted = expert_id[perm].float().unsqueeze(1)

        recv_x   = self._a2a_into(self._recv_buf, x_sorted,   send_counts, recv_counts)
        recv_eid = self._a2a_into(self._eid_recv, eid_sorted, send_counts, recv_counts)
        recv_eid = recv_eid.squeeze(1).long()

        # ★ 原理 4:按本地 expert 排成连续段,对应 DeepGEMM contiguous layout
        local_eid = recv_eid - my_first_expert          # 0..experts_per_rank-1
        expert_perm = torch.argsort(local_eid)
        recv_x_grouped = recv_x[expert_perm]
        num_recv_per_expert = torch.bincount(local_eid, minlength=self.experts_per_rank)

        h = DispatchHandle()
        h.perm, h.send_counts, h.recv_counts = perm, send_counts, recv_counts
        h.gate_sorted = gate[perm]
        h.expert_perm, h.num_recv_per_expert = expert_perm, num_recv_per_expert
        h.n_recv = recv_x.size(0)
        return recv_x_grouped, num_recv_per_expert, h

    # ★ 原理 3:combine 复用 handle,反向 a2a + 加权 + 还原,一步到位
    def combine(self, y_grouped, handle):
        # 1. 撤销 expert 分组
        y = torch.empty_like(y_grouped)
        y[handle.expert_perm] = y_grouped
        # 2. 反向 all_to_all(send/recv counts 互换),复用 send buffer
        y_back = self._a2a_into(self._send_buf, y, handle.recv_counts, handle.send_counts)
        # 3. gate 加权(对应 DeepEP combine 内部的 weighted reduction)
        y_back = y_back * handle.gate_sorted.unsqueeze(-1)
        # 4. 还原到原始 token 顺序
        out = torch.empty(y_back.size(0), y_back.size(1), dtype=y_back.dtype)
        out[handle.perm] = y_back
        return out


def run(rank, world_size, port):
    setup(rank, world_size, port)
    experts_per_rank = E // world_size
    my_experts = list(range(rank * experts_per_rank, (rank + 1) * experts_per_rank))
    n_tokens = 5

    Wg = torch.randn(E, H)
    x  = torch.randn(n_tokens, H)
    probs = torch.softmax(x @ Wg.T, dim=-1)
    gate, expert_id = probs.max(dim=-1)

    reference = torch.stack([run_expert(x[i:i+1], expert_id[i].item())[0]
                             for i in range(n_tokens)]) * gate.unsqueeze(-1)

    # ★ Buffer 创建一次(实际用时应放到进程级全局,跨 iteration 复用)
    buf = EPBuffer(dist.group.WORLD, world_size, max_tokens=n_tokens,
                   hidden=H, dtype=x.dtype)

    recv_x_grouped, num_recv_per_expert, handle = buf.dispatch(
        x, expert_id, gate, my_first_expert=my_experts[0])

    # ★ 原理 4:在连续段上跑 expert —— 每段一次 matmul = grouped GEMM 的形态
    y_grouped = torch.empty_like(recv_x_grouped)
    offset = 0
    for local_e in range(experts_per_rank):
        n = int(num_recv_per_expert[local_e])
        if n > 0:
            seg = recv_x_grouped[offset:offset + n]
            y_grouped[offset:offset + n] = run_expert(seg, my_experts[local_e])
        offset += n

    out = buf.combine(y_grouped, handle)

    diff = (out - reference).abs().max().item()
    rprint(f"experts {my_experts}, max diff {diff:.2e}")
    d = torch.tensor([diff]); dist.all_reduce(d, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\nEP global max diff = {d.item():.2e}", flush=True)
        print("OK" if d.item() < 1e-4 else "MISMATCH!", flush=True)
    cleanup()


if __name__ == "__main__":
    launch(run, world_size=4, port="29504")