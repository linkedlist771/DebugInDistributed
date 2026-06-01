import os
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger


def setup(rank: int, world_size):
    """
    rank means the number of the process, it could be
    categorized into two kinds (for example, two nodes, each with
    4 processes):
    1. local rank (0, 1, 2, 3,..)
    2. gloabl rank (0, 1, 2, .... 7)

    world_size:  total process that lanuch, ranks_per_node x node_numbers
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
        backend="gloo",
        # Don't have NCCL, so use the CPU side.
        rank=rank,
        world_size=world_size,
    )


def run(rank: int, world_size: int):
    setup(rank, world_size)
    logger.debug(f"hello from Rank {rank} / world_size {world_size}")
    dist.destroy_process_group()  # destroy all the process.


if __name__ == "__main__":
    # in the args, the rank will be passed by default.
    # nproces means the number of the processes.
    mp.spawn(run, args=(4,), nprocs=4, join=True)
