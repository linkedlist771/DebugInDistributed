import torch
import triton.language as tl
import triton
from loguru import logger

# ground truth as the golden
def vector_add_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, ret_ptr, numels, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < numels

    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)

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
    