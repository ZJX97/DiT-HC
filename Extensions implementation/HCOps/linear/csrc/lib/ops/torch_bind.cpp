#include <torch/torch.h>
#include "./torch_impl.hpp"

// Define operator `torch.ops.my_torch_op.vector_add`.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.fu2gkc7w0nrc
TORCH_LIBRARY(kpops, m)
{
    m.def("kp_linear_forward(Tensor X, Tensor W, Tensor b, Tensor Y) -> ()");
    m.def("kp_linear_backward(Tensor dY, Tensor X, Tensor W, Tensor dX, Tensor dW, Tensor db) -> ()");
    //m.def("vector_add(Tensor a, Tensor b) -> Tensor");
    //m.def("kp_layer_norm(Tensor input, int[] normalized_shape, float eps, Tensor Y, Tensor mean, Tensor rstd, Tensor? weight=None, Tensor? bias=None) -> ()");
    //m.def("kp_layer_norm_backward()");

}

// Register the implementation.
// @see
//   https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.jc288bcufw9a
TORCH_LIBRARY_IMPL(kpops, CPU, m)
{
    m.impl("kp_linear_forward", &kpops::torch_impl::kp_linear_forward);
    m.impl("kp_linear_backward", &kpops::torch_impl::kp_linear_backward);
    //m.impl("vector_add", &kpops::torch_impl::vectorAdd);
    //m.impl("kp_layer_norm", &kpops::torch_impl::kp_layer_norm);
//     m.impl("kp_layer_norm_backward", &kpops::torch_impl::kp_layer_norm_backward);
}

