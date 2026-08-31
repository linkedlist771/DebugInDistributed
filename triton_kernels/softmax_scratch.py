import torch
import triton
import triton.language as tl
from loguru import logger
DEVICE = torch.device("cuda:0")


# def naive_softmax(x: torch.Tensor):
#     # (M, N) matrix for the x:
#     x_max = x.max(dim=-1, keepdim=True).values # (M, 1)
#     z = (x - x_max).exp() 
#     return z / z.sum(dim=-1, keepdim=True)

@triton.jit
def softmax_kernel(
    out_ptr,
    in_ptr,
    in_row_stride,
    out_row_stride,
    n_rows,
    n_cols,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # triton, persistent kernel， 固定program， 每个处理多行。

    # 以row_step, 也就是program nums作为间隔来处理

    row_start = tl.program_id(axis=0)
    row_step = tl.num_programs(0)

    for row in tl.range(row_start, n_rows, step=row_step, num_stages=NUM_STAGES):


        # col_offsets = row * in_row_stride + (tl.range(BLOCK_N) < n_cols)
        row_ptr = in_ptr + row * in_row_stride
        col_offsets = tl.arange(0, BLOCK_N) # 用arange
        mask = col_offsets < n_cols

        col_values = tl.load(row_ptr + col_offsets, mask, other=float("-inf") )

        # online softmax all in the SRAM,已经是一个列了
        _max = tl.max(col_values)
        z = tl.exp((col_values - _max))
        res = z / tl.sum(z)
        tl.store(out_ptr + row * out_row_stride + col_offsets, res, mask=mask)



    
    


def softmax(x: torch.Tensor) -> torch.Tensor:

    out = torch.empty_like(x)
    n_rows, n_cols = x.shape

    # 这里其实是有问题的， 这个是每个program处理一列，
    # 所有数据都要放到SRAM里面， 如果SRAM不够的话就会rout of resources。
    # 所以后面才有online softmax
    BLOCK_N = triton.next_power_of_2(n_cols)

    # more warps to process more elements, vice versa

    num_warps = 8 if BLOCK_N >= 4096 else (4 if BLOCK_N >= 2048 else 2)

    # program 数取 SM 数的整数倍，让每个 SM 都有活干
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (min(n_rows, 4 * n_sm), )
    NUM_STAGES = 4

    softmax_kernel[grid](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        BLOCK_N=BLOCK_N,
        NUM_STAGES=NUM_STAGES,
        num_warps=num_warps
    )
    return out
    


if __name__ == "__main__":
    torch.manual_seed(0)
    # 故意用不规则形状，验证 mask
    #  raise_(OutOfResources(self.metadata.shared, max_shared, "shared memor 
    #  一个sram里面放不下
    x = torch.randn(1823, 1024, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(softmax(x), torch.softmax(x, axis=-1))
    print("ok")