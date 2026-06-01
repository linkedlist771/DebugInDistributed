"""
As for the ColumnParallelLinear, the dim is seperated in the output
dimensions.

"""


import torch.nn.functional as F
import torch
import torch.distributed as dist
from boilerplate import setup
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn


def _gather_last(x: torch.Tensor):
    ws = dist.get_world_size()
    if ws == 1:
        return x

    out = [torch.empty_like(x) for _ in range(ws)]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=-1)


def _split_last(x: torch.Tensor):
    ws, r = dist.get_world_size(), dist.get_rank()
    if ws == 1:
        return x
    c = x.size(-1) // ws
    return x[..., r * c : (r + 1) * c].contiguous()


class _Copy(torch.autograd.Function):
    """张量并行中的 copy-to-region 算子：前向恒等，反向 all-reduce 求和。

    用于张量并行块的输入端：同一份输入 x 被复制到张量并行组内的各个
    rank 上分别参与计算。前向传播时输入在各 rank 上完全相同，无需通信，
    直接透传；反向传播时，由于 x 被多个 rank 共同使用（等价于链式法则
    中一个变量分支到多处），各 rank 各自算出一份对 x 的偏导，需要通过
    all-reduce 求和得到真正的总梯度。

    与对偶算子 _Reduce（前向 all-reduce、反向恒等）配对使用，二者一进
    一出，保证切分后的并行块在数学上等价于未切分的原始层。

    Forward:
        x (Tensor): 输入张量，在张量并行组内各 rank 上一致。
        返回原样输入（identity）。

    Backward:
        g (Tensor): 上游梯度。原地在张量并行组内做 all-reduce 求和。
        返回求和后的梯度。

    Note:
        - all-reduce 为 in-place 操作，依赖已初始化的分布式进程组。
        - 通常通过 copy_to_tensor_parallel_region(x) = _Copy.apply(x) 调用。
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        return g


class _Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx, g):
        return g


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return _gather_last(x)

    @staticmethod
    def backward(ctx, g):
        return _split_last(g)


class ColumnParallelLinear(torch.nn.Module):
    def __init__(
        self, in_f: int, out_f: int, bias: bool = True, gather_output: bool = True
    ):
        super().__init__()
        ws = dist.get_world_size()
        assert out_f % ws == 0, (
            f"out_f should be divided by ws, but get out_f={out_f}, ws={ws}"
        )
        self.per = out_f // ws  # per rank out_f size
        self.gather_output = gather_output
        self.weight = torch.nn.Parameter(torch.empty(self.per, in_f))
        torch.nn.init.normal_(self.weight, std=0.02)
        self.bias = torch.nn.Parameter(torch.zeros(self.per)) if bias else None

    def forward(self, x: torch.Tensor):
        x = _Copy.apply(x)
        y = F.linear(x, self.weight, self.bias)
        return _Gather.apply(y) if self.gather_output else y


class RowParallelLinear(torch.nn.Module):
    def __init__(self, in_f, out_f, bias=True, input_is_parallel=True):
        super().__init__()
        ws = dist.get_world_size()
        assert in_f % ws == 0
        self.per = in_f // ws
        self.input_is_parallel = input_is_parallel
        self.weight = torch.nn.Parameter(torch.empty(out_f, self.per))  # local shard
        torch.nn.init.normal_(self.weight, std=0.02)
        self.bias = torch.nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        if not self.input_is_parallel:
            x = _split_last(x)
        y = F.linear(x, self.weight)  # partial sum, NO bias yet
        y = _Reduce.apply(y)  # g operator: all_reduce the partials
        if self.bias is not None:
            y = y + self.bias
        return y


class ParallelMLP(torch.nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden, ffn, gather_output=False)
        self.fc2 = RowParallelLinear(ffn, hidden, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def run(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(0)  # identical full weights on every rank
    hidden, ffn, batch = 8, 16, 4
    W1 = torch.randn(ffn, hidden)
    W2 = torch.randn(hidden, ffn)
    x = torch.randn(batch, hidden, requires_grad=True)

    ref = F.gelu(x @ W1.T) @ W2.T  # single-device reference

    mlp = ParallelMLP(hidden, ffn)
    oc = ic = ffn // world_size
    with torch.no_grad():
        mlp.fc1.weight.copy_(W1[rank * oc : (rank + 1) * oc, :])  # column shard
        mlp.fc2.weight.copy_(W2[:, rank * ic : (rank + 1) * ic])  # row shard

    par = mlp(x)
    par.sum().backward()  # exercise backward too
    if rank == 0:
        print("forward max diff:", (par - ref).abs().max().item())  # ~1e-7
    dist.destroy_process_group()


if __name__ == "__main__":
    # in the args, the rank will be passed by default.
    # nproces means the number of the processes.
    spawn(run, args=(4,), nprocs=4, join=True)
