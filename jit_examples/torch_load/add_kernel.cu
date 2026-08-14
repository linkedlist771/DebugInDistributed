#include "add_kernel.cuh"

#include <c10/cuda/CUDAException.h>

__global__ void add_kernel(
    const float* a,
    const float* b,
    float* out,
    int64_t n)
{
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");

    auto out = torch::empty_like(a);
    const int64_t n = a.numel();
    constexpr int threads = 256;

    if (n > 0) {
        const int blocks = static_cast<int>((n + threads - 1) / threads);
        add_kernel<<<blocks, threads>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            out.data_ptr<float>(),
            n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_cuda", &add_cuda, "Elementwise add (CUDA)");
}
