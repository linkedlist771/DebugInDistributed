import torch
import triton.language as tl
import triton
from loguru import logger
import os
# 必须在 import triton 之前设置，或者用 TRITON_INTERPRET=1 python s02_debug.py
os.environ["TRITON_INTERPRET"] = "0" # TRITON_INTERPRET=1， static print works
# TRITON_INTERPRET=0 device print works


# ground truth as the golden
def vector_add_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, ret_ptr, numels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # tl.static_print(f"static_print: pid :{pid}\n BLOCK_SIZE:{BLOCK_SIZE}")
    # tl.device_print("x =", x)
    # device print 不能用fstring 
    # tl.device_print(f"device_print: pid :{pid}\n BLOCK_SIZE:{BLOCK_SIZE}")
    # tl.device_print("device_print: pid :", pid)
    # tl.device_print("device_print: x_ptr :", x_ptr) # 0x720388c00000 输出一个地址

    # tl.device_print("device_print: BLOCK_SIZE:", BLOCK_SIZE)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < numels

    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)
    tl.device_print("device_print: x :", x) # 0x720388c00000 输出一个地址

    tl.store(ret_ptr + offsets, x+y, mask)    


def vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    BLOCK_SIZE = 1024

    # we could have the inplace too
    ret = torch.empty_like(x)
    numels = x.numel()
    # grid = (triton.cdiv(numels, BLOCK_SIZE), )
    grid = lambda meta: (triton.cdiv(numels, meta['BLOCK_SIZE']), )
    vector_add_kernel[grid](
        x, y, ret, numels, BLOCK_SIZE
    )


    return ret




    


if __name__ == "__main__":
    
    shape = (1026, 513)
    torch.manual_seed(42)
    x = torch.rand(size=shape, device="cuda")
    y = torch.rand(size=shape, device="cuda")
    gd = vector_add_torch(x, y)
    ret_triton = vector_add_triton(x, y)

    torch.testing.assert_close(gd, ret_triton)

    # logger.debug()
    