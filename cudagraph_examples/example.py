import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from loguru import logger
from functools import partial

DEV = "cuda"


def warmup_on_side_stream(fn: Callable, iters: int=3):
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(iters):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)



def step(static_in: torch.Tensor):
    return static_in * 2.0 + 1.0


def case01_basic_capture_replay():
    logger.critical(f"Baisc Case")
    N = 2 ** 16
    static_in = torch.zeros(N, device=DEV)

    warmup_on_side_stream(partial(step, static_in))

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # static_out = static_in * 2.0 + 1.0    
        static_out = step(static_in)

    # replay
    static_in.copy_(torch.arange(N, device=DEV, dtype=torch.float32))

    g.replay()
    torch.cuda.synchronize()
    logger.debug(f"first 5 outputs (expect 1,3,5,7,9): {static_out[:5].tolist()}")


    # replay again with different data -> reuses same kernels, no relaunch overhead
    static_in.copy_(torch.full((N,), 10.0, device=DEV))
    g.replay()
    torch.cuda.synchronize()
    logger.debug(f"after copy_(10) (expect all 21):  {static_out[:5].tolist()}" )
        
if __name__ == "__main__":
    case01_basic_capture_replay()
