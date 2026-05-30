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
    


if __name__ == "__main__":
    # in the args, the rank will be passed by default.
    # nproces means the number of the processes.
    mp.spawn(override_run, args=(4, ), nprocs=4, join=True)
    



    
    