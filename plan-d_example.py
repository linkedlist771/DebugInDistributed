"""
torch.distributed + plan-d 调试示例(dummy 版,纯 CPU)
=====================================================

和 madbg 版完全相同的场景,只是把调试器换成 plan-d(基于 rich,输出更好看,
有变量树 vt、对象 inspect、IPython magic 等)。

依赖:
    pip install plan-d torch

运行(单机起 4 个 rank):
    python plan_d_torchdist_demo.py

到达断点时,被调试的 rank 会【自动打印一条连接命令】(plan-d 自己挑空闲端口),
形如:
    plan-d connect ...        <- 照着它打印的来,不要自己猜

然后【另开一个终端】粘贴那条命令连进去。
退出调试器:输入 `exit` 或按 Ctrl+D(注意不是 q)。

注意:plan-d 文档没有明确写 set_trace 的 host/port 参数名。
本例直接用无参 set_trace(),靠它自动打印的命令连接 —— 多进程下这样最稳,
因为每个进程各自挑端口、各自打印,不会撞端口。
如果你需要【固定端口 / 从别的机器连】,见文件末尾的说明。
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
WORLD_SIZE = 4
DEBUG_RANKS = {0}         # 只调 rank 0;想调别的改这里,例如 {2}


def run_mega_routed_dummy(rank, hidden):
    """对应你的 _run_mctlass_mega_routed:这里是你真正想看的计算。"""
    routed = hidden * (rank + 1)
    scale = routed.mean()

    # ---- 只在选中的 rank 上挂调试器 ----
    if rank in DEBUG_RANKS:
        import plan_d
        logger.debug(f"[rank {rank}] plan-d breakpoint reached, "
              f"look for the 'connect' command logger.debuged below:", flush=True)
        # set_trace 会阻塞,直到你按它打印的命令连上来。
        plan_d.set_trace()

        # === 连上后停在这一行,可观察 hidden / routed / scale 等所有局部变量 ===
        # plan-d 常用命令(比 pdb 多了好看的):
        #   v / vars         列出当前局部变量
        #   vt / varstree    变量树(嵌套结构很直观)
        #   i <obj> / inspect 检查某个对象(rich 渲染)
        #   p routed         打印
        #   bt               调用栈
        #   n / s / c        下一行 / 步入 / 继续
        #   还支持 IPython magic,比如 %timeit
        #   退出用 exit 或 Ctrl+D(别用 q)
        marker = scale * 2  # noqa: F841

    # ---- 其它 rank 卡在 barrier 等 rank 0 调试完 ----
    dist.barrier()
    return routed


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # 把超时调大,否则 rank 0 在调试器里逗留期间,其它 rank 卡在 barrier 会被超时杀掉。
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=1),
    )
    logger.debug(f"[rank {rank}] init done (pid={os.getpid()})", flush=True)

    hidden = torch.ones(8) * (rank + 1)
    dist.all_reduce(hidden, op=dist.ReduceOp.SUM)
    logger.debug(f"[rank {rank}] after all_reduce, hidden[0]={hidden[0].item()}", flush=True)

    out = run_mega_routed_dummy(rank, hidden)
    logger.debug(f"[rank {rank}] out.mean={out.mean().item():.2f}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    logger.debug(f"[rank {rank}] done", flush=True)


if __name__ == "__main__":
    mp.spawn(worker, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    logger.debug("all ranks finished")


# =============================================================================
# 如果需要【固定端口】或【从别的机器连】(比如 rank 0 在远程节点上):
#
# plan-d 文档没明确写参数名,大概率是 host=/port= 这种。先试:
#     plan_d.set_trace(host="0.0.0.0", port=4444 + rank)   # 端口随 rank 区分
# 如果报 "unexpected keyword argument",说明参数名不同,退回到无参 set_trace(),
# 用它自动打印的命令连即可(只是端口是随机的)。
#
# 跑通后,把上面 if DEBUG_RANKS 那两行(import plan_d + set_trace)原样挪进
# deepseek_v2.py 第 1604 行那里就能用了。记得 export TORCH_NCCL_TIMEOUT=3600。
# =============================================================================