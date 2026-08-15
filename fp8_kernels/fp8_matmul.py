import torch

n = 4096
fp8 = torch.float8_e5m2 #torch.float8_e4m3fn

# A必须行优先，B必须列优先
a = torch.randn(n, n, device="cuda").to(fp8)
b = torch.randn(n, n, device="cuda").to(fp8).t()

c = a @ b
print(c)
# scale = torch.tensor(1.0, device="cuda")

# c = torch._scaled_mm(
#     a,
#     b,
#     scale_a=scale,
#     scale_b=scale,
#     out_dtype=torch.bfloat16,
# )

# # 兼容部分旧版PyTorch
# if isinstance(c, tuple):
#     c = c[0]

# torch.cuda.synchronize()

# print("GPU:", torch.cuda.get_device_name())
# print("Compute Capability:", torch.cuda.get_device_capability())
# print("Result:", c.shape, c.dtype)
# print("FP8 matmul: PASS")