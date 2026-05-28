"""
torch.distributed + debugpy 调试示例(dummy 版,纯 CPU)
======================================================

和 madbg/plan-d 版相同的场景,调试器换成 debugpy,用 VSCode 图形界面 attach。

与前两者的本质区别:
  - madbg/plan-d:在 set_trace 那一行停住,你在【终端】里敲命令。
  - debugpy:wait_for_client() 阻塞等 VSCode 连上;连上后你在【VSCode 里点行号
    设断点】,断开后还能重新 attach 回去(不用重跑程序)。

依赖:
    pip install debugpy torch

运行(单机起 4 个 rank):
    python debugpy_torchdist_demo.py

它会打印:
    [rank 0] debugpy listening on 0.0.0.0:5678, waiting for VSCode attach...
然后卡住,直到你在 VSCode 里按 F5 用下面的 launch.json attach 上来。

VSCode 端步骤见文件最下方,以及一起给你的 launch.json。
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
WORLD_SIZE = 4
BASE_PORT = 5678          # rank r 的 debugpy 端口 = BASE_PORT + r(端口随 rank 区分)
DEBUG_RANKS = {0}         # 只调 rank 0;想调别的改这里,例如 {2}


def run_mega_routed_dummy(rank, hidden):
    """对应你的 _run_mctlass_mega_routed。"""
    routed = hidden * (rank + 1)
    scale = routed.mean()

    # ↓↓↓ 在 VSCode 里,把断点点在下面这几行的行号左侧(红点)↓↓↓
    marker = scale * 2          # <- 在这一行设断点试试
    result = routed + marker    # <- 或这一行
    # ↑↑↑ debugpy 是在 VSCode 里点行号设断点,不是在代码里写 set_trace ↑↑↑

    dist.barrier()              # 其它 rank 卡在这里等 rank 0 调试完
    return result


def maybe_start_debugpy(rank):
    """只在选中的 rank 上启动 debugpy 并阻塞等待 VSCode attach。"""
    if rank not in DEBUG_RANKS:
        return
    import debugpy
    port = BASE_PORT + rank
    debugpy.listen(("0.0.0.0", port))   # 监听所有网卡,允许远程 attach
    logger.debug(f"[rank {rank}] debugpy listening on 0.0.0.0:{port}, "
          f"waiting for VSCode attach...", flush=True)
    debugpy.wait_for_client()           # 阻塞,直到 VSCode 连上
    logger.debug(f"[rank {rank}] VSCode attached.", flush=True)
    # 注:debugpy.listen 每个进程只能调一次。这里在 init 之后、forward 之前调一次最稳。


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(hours=1),     # 调大,避免 rank 0 调试时其它 rank 在 barrier 超时
    )
    logger.debug(f"[rank {rank}] init done (pid={os.getpid()})", flush=True)

    # 在进入计算前 attach,这样你来得及在 VSCode 里设断点
    maybe_start_debugpy(rank)

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
# VSCode 端步骤:
#
# 1. 把同目录下的 launch.json 放到 你的工程/.vscode/launch.json
#    (如果本地连不到服务器 IP,见 launch.json 里的 SSH 隧道说明)
# 2. 终端跑 python debugpy_torchdist_demo.py,等它打印 "waiting for VSCode attach..."
# 3. VSCode 里在 run_mega_routed_dummy 的 marker 那行点一个断点(行号左边红点)
# 4. 按 F5,选择 "Attach sglang rank0" 这个配置
# 5. 程序会跑到你设的断点停下,左侧能看 hidden / routed / scale 的值,
#    可以单步、悬停看变量、在 Debug Console 里敲 routed.shape 等表达式
# 6. 调完点断开;想再看可以再按 F5 重新 attach(进程没退就行)
#
# 搬到真实 sglang:把 maybe_start_debugpy(rank) 的逻辑放到模型 forward 之前
# 调一次即可,然后在 deepseek_v2.py 里直接点行号设断点。
# 关键:launch.json 里 "justMyCode": false 必须有,否则断点进不去库代码/模型内部。
# 真实环境记得 export TORCH_NCCL_TIMEOUT=3600。
# =============================================================================