#pragma once

#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
