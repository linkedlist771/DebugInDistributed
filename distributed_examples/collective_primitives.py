from boilerplate import setup
import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger
from utils import setup_logger


def override_run(rank: int, world_size: int):
    setup_logger(rank, world_size)

    setup(rank, world_size)

    ## 1. All reduce: every rank contributes and gets the reduced results
    t = torch.tensor([ rank + 1.0])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    logger.debug(f"all_reduce sum results t: {t.item()}")

    ## 2. All gather: collect each rank's shard data into the full tensor, 
    # on each rank
    g = torch.tensor([ rank + 1.0])
    buf = [torch.zeros(1) for _ in range(world_size)]
    # buf 里的顺序由 rank 编号唯一决定,而且是确定性的——和谁先到、谁算得快、网络快慢都无关。
    # buf[i] == 来自 rank i 的那个 g 
    #  buffer 里面的对应的idx就对应global rank的哪个数据

    dist.all_gather(buf, g)
    full = torch.cat(buf)
    logger.debug(f"All gather results: full: {full}")

    ## 3. Reduce scatter: 

    # reduce_scatter 做两件事:先归约,再分发。它把所有 rank 上"同一位置"的分片逐元素归约(这里 op=SUM 就是相加),然后把第 j 个归约结果只发给 rank j。所以它和 all_reduce 的关键区别是:每个 rank 最后只拿到结果的"一片",而不是完整结果。
    # 只拿到部分
    # shards = [torch.tensor([float(rank)]) for _ in range(world_size)]
    # out = torch.zeros(1)
    # dist.reduce_scatter(out, shards, op=dist.ReduceOp.SUM)

    # logger.debug(f"Got out from reduce scatter out:\n{out}")

    shards = [torch.tensor([float(rank * world_size + i + 1)]) for i in range(world_size)]
    out = torch.zeros(1)
    dist.reduce_scatter(out, shards, op=dist.ReduceOp.SUM)

    logger.debug(f"Got out from reduce scatter out:\n{out}")

    ## 4. broadcast， one rank's tensor copied to all.

    b = torch.tensor([99.0]) if rank == 0 else torch.tensor([0.0])

    dist.broadcast(b, src=0) # 从哪个rank发送
    logger.debug(f"got broadcasted b: {b}")
    


if __name__ == "__main__":
    # in the args, the rank will be passed by default.
    # nproces means the number of the processes.
    mp.spawn(override_run, args=(4, ), nprocs=4, join=True)
    

