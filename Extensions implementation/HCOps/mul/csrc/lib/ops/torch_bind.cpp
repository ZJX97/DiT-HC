#include <torch/torch.h>
#include "./torch_impl.hpp"

// Define operator `torch.ops.my_torch_op.vector_add`.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.fu2gkc7w0nrc
TORCH_LIBRARY(w2kpops, m) {
	m.def("modulate(Tensor a, Tensor sh, Tensor sc) -> Tensor");
	m.def("modulate_impl(Tensor a, Tensor sh, Tensor sc) -> Tensor[]");
	m.def("modulate_backward_impl(Tensor a, Tensor sc, Tensor grad) -> Tensor[]");

	m.def("mla(Tensor x, Tensor gate, Tensor y) -> Tensor");
	m.def("mla_impl(Tensor a, Tensor gate, Tensor y) -> Tensor[]");
	m.def("mla_backward_impl(Tensor a, Tensor gate, Tensor y) -> Tensor[]");
}

// Register the implementation.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.jc288bcufw9a
TORCH_LIBRARY_IMPL(w2kpops, CPU, m) {
	m.impl("modulate_impl", &kpops::torch_impl::modulate_forward_impl);
	m.impl("modulate_backward_impl", &kpops::torch_impl::modulate_backward_impl);

	m.impl("mla_impl", &kpops::torch_impl::mla_forward_impl);
	m.impl("mla_backward_impl", &kpops::torch_impl::mla_backward_impl);
}

TORCH_LIBRARY_IMPL(w2kpops, Autograd, m) {
	m.impl("modulate_impl", &kpops::torch_impl::modulate_impl_autograd);
	m.impl("mla_impl", &kpops::torch_impl::mla_impl_autograd);

}

TORCH_LIBRARY_IMPL(w2kpops, CompositeImplicitAutograd,m) {
	m.impl("modulate", &kpops::torch_impl::modulate);	
	m.impl("mla", &kpops::torch_impl::mla);	
}