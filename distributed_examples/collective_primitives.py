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
    


if __name__ == "__main__":
    # in the args, the rank will be passed by default.
    # nproces means the number of the processes.
    mp.spawn(override_run, args=(4, ), nprocs=4, join=True)
    



    
    