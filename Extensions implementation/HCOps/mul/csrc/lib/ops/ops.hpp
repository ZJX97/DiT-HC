#include <cstddef>
#include <cstdint>
#include <torch/torch.h>

#ifndef _SVE_DF_
#define _SVE_DF_
#define SF svfloat32_t
#define SO svbool_t
#endif

namespace kpops {
    void modulateKernel(const torch::Tensor& x, const torch::Tensor& sh, const torch::Tensor& sc, torch::Tensor& result);
    void modulateBackwardKernel(const torch::Tensor& x, const torch::Tensor& sc,torch::Tensor& grad_x,torch::Tensor& grad_sh, torch::Tensor& grad_sc,const torch::Tensor& grad_output);
    void mlaKernel(const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& y, torch::Tensor& result);
    void mlaBackwardKernel(const torch::Tensor& gate, const torch::Tensor& y,const torch::Tensor& grad_output, torch::Tensor& grad_gate, torch::Tensor& grad_y);
}  // namespace kpops::cpu

