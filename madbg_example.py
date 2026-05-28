"""
torch.distributed + madbg 调试示例(dummy 版,纯 CPU)
=====================================================

模拟你 sglang 里 `_run_mctlass_mega_routed` 的真实场景:
多个 rank、有 collective(all_reduce)、有 dist.barrier(),
只在 rank 0 上挂调试器,其它 rank 在 barrier 处等待。

用 gloo backend + CPU,所以不需要 GPU,任何 Linux 机器都能直接跑。

依赖:
    pip install madbg torch

运行(单机起 4 个 rank):
    python madbg_torchdist_demo.py

终端会打印形如:
    [rank 0] madbg listening -> madbg connect 127.0.0.1 3513

然后【另开一个终端】执行那条命令连进去(IPython 界面,有高亮/补全):
    madbg connect 127.0.0.1 3513
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD_SIZE = 4
BASE_PORT = 3513          # rank r 的调试端口 = BASE_PORT + r
DEBUG_RANKS = {0}         # 只调 rank 0;想调别的就改这里,例如 {2}

# madbg connect 127.0.0.1 3513 
# connect to the dbeug...

def run_mega_routed_dummy(rank, hidden):
    """对应你的 _run_mctlass_mega_routed:这里是你真正想看的计算。"""
    # 造点假的 "MoE" 中间量,方便断点里观察
    routed = hidden * (rank + 1)
    scale = routed.mean()

    # ---- 只在选中的 rank 上挂调试器 ----
    if rank in DEBUG_RANKS:
        import madbg
        port = BASE_PORT + rank                  # 关键:端口随 rank 区分,避免冲突
        print(f"[rank {rank}] madbg listening -> madbg connect 127.0.0.1 {port}",
              flush=True)
        # ip='0.0.0.0' 允许从别的机器连;本机调试用 '127.0.0.1' 也行。
        # set_trace 会阻塞,直到你 madbg connect 连上来。
        madbg.set_trace(ip="0.0.0.0", port=port)

        # === 连上后停在这一行,可观察 hidden / routed / scale 等所有局部变量 ===
        # 试试:
        #   hidden.shape
        #   pp routed
        #   routed[:5]
        #   scale.item()
        #   n / s / c   下一行 / 步入 / 继续    (别乱按 q)
        marker = scale * 2  # noqa: F841  给你一行能 step 过去的代码

    # ---- 其它 rank 会先跑到这里,卡在 barrier 等 rank 0 调试完 ----
    dist.barrier()   # 注意:rank 0 在调试器里逗留期间,这里其它 rank 在等待

    return routed


def worker(rank, world_size):
    # 单机多进程:都连同一个 master
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # 关键:把超时调大,否则 rank 0 在调试器里待超过默认时长,
    # 其它 rank 卡在 barrier 会被超时杀掉(真实场景里就是 NCCL timeout)。
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=1),
    )
    print(f"[rank {rank}] init done (pid={os.getpid()})", flush=True)

    # 造个假的 hidden 张量,做个真实的 collective 让场景更像
    hidden = torch.ones(8) * (rank + 1)
    dist.all_reduce(hidden, op=dist.ReduceOp.SUM)   # 所有 rank 求和
    print(f"[rank {rank}] after all_reduce, hidden[0]={hidden[0].item()}", flush=True)

    out = run_mega_routed_dummy(rank, hidden)
    print(f"[rank {rank}] out.mean={out.mean().item():.2f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] done", flush=True)


if __name__ == "__main__":
    # spawn 出 WORLD_SIZE 个进程,等价于单机 torchrun --nproc_per_node=WORLD_SIZE
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    print("all ranks finished")