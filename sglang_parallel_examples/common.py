
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable
from loguru import logger

def setup(rank: int, world_size: int, port: str = "29500", seed: int = 0):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # Same seed on every rank => every rank materializes the *same* "full"
    # weights, then slices out its own shard. This is the standard trick that
    # lets us compare the sharded result against a single-device reference.
    torch.manual_seed(seed)


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def launch(fn: Callable, world_size: int, port: str = "29500"):
    """mp.spawn launches `world_size` copies of fn(rank, world_size, port)."""
    mp.spawn(fn, args=(world_size, port), nprocs=world_size, join=True)


def rprint(*args):
    msg = " ".join(str(a) for a in args)
    ws = dist.get_world_size()
    gathered = [None] * ws
    dist.all_gather_object(gathered, msg)
    if dist.get_rank() == 0:
        for r, m in enumerate(gathered):
            logger.debug(f"[rank {r}] {m}", flush=True)



def head(title: str):
    if dist.get_rank() == 0:
        logger.debug("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)  # flush: avoid mp dup
    dist.barrier()
