"""
Expert Parallel Flow:
    route locally ─► sort tokens by destination rank
                  ─► all_to_all the per-dest counts   (so peers size their buffers)
                  ─► all_to_all the tokens            (DISPATCH)
                  ─► run my local experts on what I received
                  ─► all_to_all the results back      (COMBINE)
                  ─► unsort, scale by gate weight
"""


import torch
import torch.distributed as dist
import torch.nn.functional as F

from common import cleanup, launch, rprint, setup



H = 16  # hidden size
I = 32  # expert hidden size
E = 8   # total experts


def expert_weights(eid: int):
    """
    Init the weight for a specific expert.
    """

    # 是在跟 PyTorch nn.Linear 的权重布局约定保持一致。nn.Linear(in, out) 内部存的权重形状是 (out, in),前向计算是 x @ weight.T(也就是 F.linear)
    g = torch.Generator().manual_seed(2000 + eid)
    return torch.randn(I, H, generator=g), torch.randn(H, I, generator=g)


def a2a_counts(send_counts):
    send = send_counts.float()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send)
        #     >>> # xdoctest: +SKIP("Undefined rank")
        # >>> input = torch.arange(4) + rank * 4
        # >>> input
        # tensor([0, 1, 2, 3])     # Rank 0
        # tensor([4, 5, 6, 7])     # Rank 1
        # tensor([8, 9, 10, 11])   # Rank 2
        # tensor([12, 13, 14, 15]) # Rank 3
        # >>> output = torch.empty([4], dtype=torch.int64)
        # >>> dist.all_to_all_single(output, input)
        # >>> output
        # tensor([0, 4, 8, 12])    # Rank 0
        # tensor([1, 5, 9, 13])    # Rank 1
        # tensor([2, 6, 10, 14])   # Rank 2
        # tensor([3, 7, 11, 15])   # Rank 3
    return recv.long()


def a2a_rows(x, in_splits, out_splits):
    """all_to_all_single over dim-0 row groups with variable split sizes."""
    out = torch.empty(int(out_splits.sum()), x.size(1), dtype=x.dtype)
    dist.all_to_all_single(
        out, x.contiguous(),
        output_split_sizes=out_splits.tolist(),
        input_split_sizes=in_splits.tolist(),
    )
    return out






def run_expert(x: torch.Tensor, eid) -> torch.Tensor:
    w_in, w_out = expert_weights(eid)
    act_res = F.silu(x @ w_in.T) 
    return act_res @ w_out.T



def run(rank: int, world_size: int, port: str):
    setup(rank, world_size, port)
    assert E % world_size == 0
    experts_per_rank = E // world_size
    my_experts = list(range(rank * experts_per_rank, (rank + 1) * experts_per_rank))

    n_tokens = 5
    
    # router weight, each rank shares the same router weight.
    Wg = torch.randn(E, H)
    # input tensor, with DP, each rank has a different input tensor
    x = torch.randn(n_tokens, H) 

    ## 1. routing(top-1)
    probs = torch.softmax(x @ Wg.T, dim=-1) # (n, E) probabilities

    # top 1 expert and its weight
    gate, expert_id = probs.max(dim=-1) 
    # # top-k 才需要的一步(本代码没有,因为是 top-1)
    # gate = gate / gate.sum(dim=-1, keepdim=True)   # 0.42, 0.20 → 0.677, 0.323
    
    ## reference:
    reference = torch.stack([run_expert(x[i:i+1], expert_id[i].item())[0]
                             for i in range(n_tokens)]) * gate.unsqueeze(-1)
    
    ## 2. EP dispatch
    dest = expert_id // experts_per_rank  # 这些expert_id都在那些rank上， 
    # 因为现在给rank分配expert id是按照experts_per_rank来的
    # torch.argsort(dest) 是返回排序的顺序而不是排序完的结果
    # perm = torch.argsort(dest) —— 这是最关键的一步。all_to_all_single 要求输入张量在第 0 维按目的 rank 连续排列(先放所有发给 rank 0 的行,再放发给 rank 1 的……)。但此刻 token 是乱序的,所以用 argsort 求出"把 dest 排好序需要的索引顺序"。
    # dest=[1,0,3,1,0] 的 argsort = [1,4,0,3,2](把 dest=0 的 token1、token4 排前面,然后 dest=1 的 token0、token3,最后 dest=3 的 token2)。


    # 按照destination 来group这些 tokens
    perm = torch.argsort(dest)
    # 一一对应
    x_sorted = x[perm]
    eid_sorted = expert_id[perm]
    gate_sorted = gate[perm]
    send_counts = torch.bincount(dest, minlength=world_size)
    #     `torch.bincount(dest)` 返回一个数组,**第 `i` 个位置 = 数值 `i` 在 `dest` 里出现的次数**。

    # 用前面那组 `dest = [1, 0, 3, 1, 0]`:

    # ```
    # 统计:  0 出现 2 次
    #         1 出现 2 次
    #         2 出现 0 次
    #         3 出现 1 次

    # bincount → [2, 2, 0, 1]
    #             ↑  ↑  ↑  ↑
    #           值0 值1 值2 值3 的次数
    # ```

    # 注意结果的**下标本身就是被统计的数值**。所以 `result[i]` 读作"数值 `i` 出现了几次"。这恰好是这里需要的:`dest` 装的是目的 rank 编号,数完之后 `send_counts[i]` 就是"要发给 rank `i` 的 token 数"。

    # ## `minlength=world_size` 是干嘛的

    # 不加这个参数时,`bincount` 的输出长度是 **`max(dest) + 1`**——它只数到出现过的最大值为止。

    # 这就有个隐患:假设某个 rank 的 token 谁都没分到最后那张卡,比如 `dest = [1, 0, 1, 0]`(最大值是 1),那么:

    # ```python
    # torch.bincount([1,0,1,0])                  → [2, 2]        # 长度只有 2!
    # torch.bincount([1,0,1,0], minlength=4)     → [2, 2, 0, 0]  # 强制补到长度 4
    # ```
    recv_counts = a2a_counts(send_counts)

    recv_x = a2a_rows(x_sorted, send_counts, recv_counts)
    
    recv_eid = a2a_rows(eid_sorted.float().unsqueeze(1), send_counts, recv_counts).squeeze(1).long()

    ## 3. run local experts for the received tokens
    y = torch.empty_like(recv_x)
    for eid in my_experts:
        m = recv_eid == eid 
        if m.any():
            y[m] = run_expert(recv_x[m], eid)

    ## 4. Combine

    y_back = a2a_rows(y, recv_counts, send_counts)
    y_back = y_back * gate_sorted.unsqueeze(-1)
    out = torch.empty_like(y_back)
    out[perm] = y_back
    diff = (out - reference).abs().max().item()
    rprint(f"owns experts {my_experts}, routed dests {dest.tolist()}, max diff {diff:.2e}")
    d = torch.tensor([diff])
    dist.all_reduce(d, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\nEP (ep_size={world_size}, {experts_per_rank} experts/rank): "
              f"global max diff = {d.item():.2e}", flush=True)
        print("OK" if d.item() < 1e-4 else "MISMATCH!", flush=True)
    cleanup()


if __name__ == "__main__":
    launch(run, world_size=4, port="29504")
