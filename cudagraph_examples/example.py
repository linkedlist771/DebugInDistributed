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


def case02_make_graphed_callables():
    logger.critical(f"make_graphed_callables")
    B, D = 32, 256
    model = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D)).to(DEV)
    sample = torch.randn(B, D, device=DEV)

    graphed = torch.cuda.make_graphed_callables(model, (sample,), num_warmup_iters=3)

    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    for it in range(3):
        x = torch.randn(B, D, device=DEV)
        target = torch.randn(B, D, device=DEV)
        opt.zero_grad(set_to_none=True)
        out = graphed(x)                 # forward graph
        loss = F.mse_loss(out, target)
        loss.backward()                  # backward graph
        opt.step()
        torch.cuda.synchronize()
        logger.debug(f"  iter {it}: loss={loss.item():.4f}")  



def case03_torch_compile_reduce_overhead():
    """torch-integration.html -> 'Automatic CUDA Graphs with torch.compile()'.
    Zero manual capture code; the inductor backend graphs compatible regions
    (CUDAGraph Trees). Requires inductor/triton — may be unavailable on some backends.
    """
    try:
        @torch.compile(mode="reduce-overhead")
        def f(x):
            return (x * 2.0 + 1.0).relu()

        x = torch.randn(1 << 16, device=DEV)
        for _ in range(3):                       # first call compiles, then graphs
            y = f(x)
        torch.cuda.synchronize()
        print("compiled+graphed output first 5:", y[:5].tolist())
    except Exception as e:
        print(f"torch.compile path unavailable on this backend: {type(e).__name__}: {e}")

    

def case04_copy_vs_reassign():
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

    # WRONG: rebinding the name. Graph still reads the ORIGINAL buffer (holds 0..7).
    static_in = torch.full((N,), 999.0, device=DEV)   # new tensor, new address
    g.replay(); torch.cuda.synchronize()
    logger.debug(f"after rebind to 999 -> {static_out.tolist()[:5]}"
          f" (still 100..107 — rebind ignored!)")


# CASE 06 — ASYNC: CPU<->GPU sync during capture is FORBIDDEN
# --------------------------------------------------------------------------- #
def case06_no_cpu_sync_during_capture():
    """constraints.html -> 'No Host-Device Synchronization' / best-practices 'ASYNC'.
    .item()/.cpu()/print(tensor) all block on the GPU -> capture fails.
    Fix: keep the value on-device and branch with torch.where (Case 08).
    """
    x = torch.randn(1024, device=DEV)

    def step():
        return x * 2.0

    warmup_on_side_stream(step)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            y = x * 2.0
            if y.sum().item() > 0:  
                y = y + 1.0
    except:
        from traceback import format_exc
        logger.error(format_exc())


    # sync-free version: no host sync at all
    warmup_on_side_stream(step)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        y2 = (x * 2.0) + 1.0
    g2.replay(); torch.cuda.synchronize()
    print("sync-free capture OK, first value:", y2[0].item())




# --------------------------------------------------------------------------- #
# CASE 07 — GPU-ONLY: CPU code runs ONCE at capture, never on replay
# --------------------------------------------------------------------------- #
def case07_cpu_code_not_captured():
    """constraints.html -> 'CPU Code Is Not Captured' / best-practices 'GPU-ONLY'.
    Python counters, list.append, logging inside the capture region execute exactly
    once (at capture) and are eliminated from replay.
    """
    # banner(7, "GPU-ONLY: Python-side bookkeeping is not replayed")
    static_in = torch.zeros(8, device=DEV)
    counter = {"n": 0}
    captured = []

    def step():
        return static_in + 1.0

    warmup_on_side_stream(step)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = static_in + 1.0    # CPU side code will not be caputred
        counter["n"] += 1               # CPU code: runs once, now
        captured.append(static_out)     # CPU code: runs once, now

    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    print(f"counter after 5 replays = {counter['n']}  (still 1 — CPU code not replayed)")
    print(f"len(captured list)       = {len(captured)}  (still 1)")



# --------------------------------------------------------------------------- #
# CASE 09 — SELF-CONTAINED STREAM CAPTURE (the routed-path crash analog)
# --------------------------------------------------------------------------- #
def case09_stream_fork_join():
    """constraints.html -> 'Self-Contained Stream Capture' (fork-join) + 'No Default Stream'.
    This is the direct analog of the MegaMoE routed-path capture crash.
      9a CORRECT : fork to a side stream, then JOIN back before capture ends.
      9b UNJOINED: fork but never join -> error 904 (mcErrorStreamCaptureUnjoined).
      9c IMPLICIT: let an op touch the legacy/default stream during capture
                   -> error 906 ("legacy stream depends on capturing blocking stream").
    """
    N = 1024
    x = torch.ones(N, device=DEV)

    # ---- 9a: correct fork-join -------------------------------------------- #
    def step_a():
        side = torch.cuda.Stream()
        a = x * 2.0
        side.wait_stream(torch.cuda.current_stream())   # FORK
        with torch.cuda.stream(side):
            b = a + 100.0
        torch.cuda.current_stream().wait_stream(side)    # JOIN
        return b - 1.0

    warmup_on_side_stream(step_a)
    g_a = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    with torch.cuda.graph(g_a):
        a = x * 2.0
        side.wait_stream(torch.cuda.current_stream())    # FORK from capture stream
        with torch.cuda.stream(side):
            b = a + 100.0
        torch.cuda.current_stream().wait_stream(side)    # JOIN back -> self-contained
        out_a = b - 1.0
    g_a.replay(); torch.cuda.synchronize()
    print(f"9a fork-join OK -> {out_a[0].item()} (expect (1*2+100)-1 = 101)")

    # # ---- 9b: fork without join -> 904 ------------------------------------- #
    # g_b = torch.cuda.CUDAGraph()
    # side_b = torch.cuda.Stream()
    # try:
    #     with torch.cuda.graph(g_b):
    #         a = x * 2.0
    #         side_b.wait_stream(torch.cuda.current_stream())   # FORK
    #         with torch.cuda.stream(side_b):
    #             _ = a + 5.0                                    # work on side stream
    #         # NO join back -> side stream left dangling at end-capture
    # except Exception as e:
    #     logger.error(e)
    # ---- 9c: op on the legacy/default stream during capture -> 906 -------- #
    g_c = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_c):
        a = x * 2.0
        with torch.cuda.stream(torch.cuda.default_stream()):  # legacy stream!
            _ = a + 1.0                                       # implicit dependency



if __name__ == "__main__":
    case09_stream_fork_join()
