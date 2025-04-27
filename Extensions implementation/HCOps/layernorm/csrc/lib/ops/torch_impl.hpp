#pragma once

#include <torch/torch.h>
#include <c10/util/ArrayRef.h>
namespace kpops::torch_impl
{
void kp_linear_forward(const torch::Tensor& X, const torch::Tensor& W, const torch::Tensor& b, torch::Tensor& Y);
void kp_linear_backward(const torch::Tensor& dY, const torch::Tensor& X, const torch::Tensor& W, torch::Tensor& dX, torch::Tensor& dW, torch::Tensor& db);
//
}  // namespace kpops::torch_impl
