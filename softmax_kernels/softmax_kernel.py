import torch
import triton
import triton.language as tl
from loguru import logger
DEVICE = torch.device("cuda:0")


def naive_softmax(x: torch.Tensor):
    # (M, N) matrix for the x:
    x_max = x.max(dim=-1, keepdim=True).values # (M, 1)
    z = (x - x_max).exp() 
    return z / z.sum(dim=-1, keepdim=True)


def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3,
                        device=DEVICE):
    # assert type(size) is tuple and 
    torch.manual_seed(0)
    x = torch.randn(size=size, device=DEVICE)
    z_tri = naive_softmax(x)
    z_ref = torch.softmax(x, dim=-1)
    z_triton = softmax(x)
    torch.testing.assert_close(z_tri, z_ref, atol=atol,
                               rtol=rtol)

    torch.testing.assert_close(z_tri, z_triton, atol=atol,
                               rtol=rtol)


properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
logger.debug(f"properties:{properties}")


TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
warpSize = properties["warpSize"]
NUM_REGES = properties["max_num_regs"]
multiprocessor_count = properties["multiprocessor_count"]

def softmax(x: torch.Tensor):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4
    # give more warps for the ...
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16 

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    y = torch.empty_like(x)

    # warmup
    
    kernel = _softmax_kernel.warmup(
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )

    kernel._init_handles()
    n_regs_per_program = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared

    reg_occupancy = NUM_REGES // (n_regs_per_program * 
                                  warpSize * num_warps)

    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program 


    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(multiprocessor_count * programs_per_sm, n_rows)

    grid = (num_programs, 1, 1)

    # x.shape : (M, N)
    # it means how many steps it will take 
    # to the next element along this axis
    # x.stride() => (N, 1)

    # z: (N, B, D)
    # z.stride(): (N*D, D, 1) => continuous
    # when applying the view, it adjust the stride
    # 
    assert x.is_contiguous()
    # 直接启动 warmup 返回的 CompiledKernel 时,只接受位置参数:
    # constexpr 参数(BLOCK_SIZE/num_stages)在 launcher 里仍占一个
    # 占位槽位,必须按位置补上;num_warps 不是 kernel 参数,不用传
    kernel[grid](   x, y,
                 x.stride(0),
                 y.stride(0),
                 n_rows, n_cols,
                 BLOCK_SIZE, num_stages,
            )

    return y
    

    
    


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    x_row_stride,
    y_row_stride,
    n_rows,
    n_clos,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,

):
    # each pid handles on row
    pid = tl.program_id(axis=0)

    row_step = tl.num_programs(0)


    for row_idx in tl.range(pid, n_rows, row_step,
                            num_stages=num_stages):
        # when n_rows> row_step
        # row_idx have to jump to the next ...
        # num_stages can do more things in parallel
        
        row_start_ptr = x_ptr + row_idx * x_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
            #    offset        + offsets
        x_ptrs = row_start_ptr + col_offsets
        # we only care about the rows within
        # the limit of the all column element
        mask = col_offsets < n_clos

        # we only read it once from the memory.
        row = tl.load(x_ptrs, mask=mask, other=float("-inf"))


        ## calculation:
        # row only has one dimension.
        row_minus_max = row - tl.max(row, axis=0) 
        # (BLOCK_SIZE) - (1) -> (BLOCK_SIZE)
        # exp2 means 以2为底数， 用换底公式
        numerator = tl.exp2(row_minus_max * 1.4426950408889634)

        denominator = tl.sum(numerator, axis=0) # shape (1)

        softmax = numerator / denominator
        # (BLOCK_SIZE) / (1) -> (BLOCK_SIZE)
        
        y_row_start_ptr = y_ptr + row_idx * y_row_stride

        # and only one store.
        tl.store(y_row_start_ptr + col_offsets, softmax, mask=mask)
    # offset = pid * BLOCK_SIZE 

    



if __name__ == "__main__":
    shape = (1823, 512)
    test_softmax_kernel(size=shape)

