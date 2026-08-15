import torch
import torch.nn.functional as F

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

print("F.scaled_mm:", hasattr(F, "scaled_mm"))
print("torch._scaled_mm:", hasattr(torch, "_scaled_mm"))
print("FP8 dtype:", hasattr(torch, "float8_e4m3fn"))