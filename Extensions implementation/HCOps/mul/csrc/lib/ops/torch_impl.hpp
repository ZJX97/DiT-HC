#include <torch/torch.h>

namespace kpops::torch_impl {
	auto modulate(const torch::Tensor& x, const torch::Tensor& sh, const torch::Tensor& sc) -> torch::Tensor;
	auto modulate_forward_impl(const torch::Tensor& x, const torch::Tensor& sh, const torch::Tensor& sc) -> torch::autograd::variable_list;
	auto modulate_backward_impl(const torch::Tensor& x,const torch::Tensor& sc, const torch::Tensor& grad_output) -> torch::autograd::variable_list;
	auto modulate_impl_autograd(const torch::Tensor& x, const torch::Tensor& sh, const torch::Tensor& sc) -> torch::autograd::variable_list;

	auto mla(const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& y) -> torch::Tensor;
	auto mla_forward_impl(const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& y) -> torch::autograd::variable_list;
	auto mla_backward_impl(const torch::Tensor& gate,const torch::Tensor& y, const torch::Tensor& grad_output) -> torch::autograd::variable_list;
	auto mla_impl_autograd(const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& y) -> torch::autograd::variable_list;

}  // namespace kpops::torch_impl

