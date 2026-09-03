# from tilelang.metal.pipeline import PassPipeline
import torch
import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1) 
    # # (M, N) matrix for the x:
    # x_max = x.max(dim=-1, keepdim=True).values  # (M, 1)
    # z = (x - x_max).exp()
    # return z / z.sum(dim=-1, keepdim=True)




@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    rows,
    cols,
    BLOCK_N_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for _row_start in tl.range(pid, rows, num_programs, num_stages=3):
        row_start =  _row_start * input_row_stride + input_ptr

        running_max = float("-inf")
        running_sum = 0.0

        for col_start in range(0, cols, BLOCK_N_SIZE):
            col_offsets = tl.arange(0, BLOCK_N_SIZE) + col_start 
            mask = col_offsets < cols
            col_values = tl.load(row_start + col_offsets, mask=mask, other=float("-inf"))
            
            # 迭代公式
            new_running_max = tl.maximum(running_max, tl.max(col_values, axis=0))
            running_sum = running_sum * tl.exp(running_max - new_running_max) + \
            tl.sum(tl.exp(col_values -new_running_max), axis=0)
            running_max = new_running_max

        # 已经获取得到了最大的和sum了， 再扫描一次进行load就行了
        for col_start in range(0, cols, BLOCK_N_SIZE):
            col_offsets = tl.arange(0, BLOCK_N_SIZE) + col_start 
            mask = col_offsets < cols
            col_values = tl.load(row_start + col_offsets, mask=mask, other=float("-inf"))

            tl.store(output_ptr + _row_start * input_row_stride + col_offsets, tl.exp(col_values-running_max) / running_sum,
            mask=mask)
            


        
        
    # rowinput_row_stride * pid
    

def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    BLOCK_N_SIZE = 1024
    # BLOCK_N_SIZE = 

    M, N = x.shape

    out = torch.empty_like(x)
    

    # grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N_SIZE"]), )
    grid = (256,)  # 每行一个 program
    # grid = (M,)  # 每行一个 program

    softmax_kernel[grid](x, out, x.stride(0), M, N, BLOCK_N_SIZE)
    return out
    
if __name__ == "__main__":
    x = torch.randn(51200, 8192, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(triton_softmax(x), torch.softmax(x, -1),
                               rtol=1e-4, atol=1e-4)
    print("ok")
    