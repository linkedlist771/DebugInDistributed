import os
import shutil
from pathlib import Path


# The pip CUDA toolkit exposes nvcc through a symlink in ~/.local/bin. Resolve
# that symlink so PyTorch can find the toolkit headers next to the real nvcc.
nvcc = shutil.which("nvcc")
if "CUDA_HOME" not in os.environ and nvcc is not None:
    os.environ["CUDA_HOME"] = str(Path(nvcc).resolve().parent.parent)

import torch
from torch.utils.cpp_extension import load

my_kernel = load(
    name="my_kernel",
    sources=[str(Path(__file__).with_name("add_kernel.cu"))],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)
a = torch.arange(1024, device="cuda", dtype=torch.float32)
b = torch.arange(1024, device="cuda", dtype=torch.float32)
out = my_kernel.add_cuda(a, b)

torch.testing.assert_close(out, a + b)
print(out)
