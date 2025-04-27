#include "../ops.hpp"
#include "../torch_impl.hpp"
#include<cstdio>
#include<iostream>
#include<cstdlib>
#include<time.h>

namespace kpops::torch_impl {
    torch::autograd::variable_list modulate_forward_impl(const torch::Tensor& x, 
        const torch::Tensor& sh, const torch::Tensor& sc) {
            torch::Tensor result = torch::empty_like(x);
            kpops::modulateKernel(x, sh, sc, result);
		return {result};
    }
    torch::autograd::variable_list modulate_backward_impl(const torch::Tensor& x,
        const torch::Tensor& sc, const torch::Tensor& grad_output) {
		
        torch::Tensor grad_x = torch::empty_like(x);
        torch::Tensor grad_sh = torch::zeros_like(sc);
        torch::Tensor grad_sc = torch::zeros_like(sc);

        kpops::modulateBackwardKernel(x ,sc, grad_x,grad_sh, grad_sc, grad_output);

        return {grad_x, grad_sh, grad_sc};
    }
	
	class Modulate : public torch::autograd::Function<Modulate> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, 
        const torch::Tensor& x, const torch::Tensor& sh, const torch::Tensor& sc) {
            at::AutoDispatchBelowADInplaceOrView guard;
			static auto op = torch::Dispatcher::singleton()
                .findSchemaOrThrow("w2kpops::modulate_impl", "")
                .typed<decltype(modulate_forward_impl)>();
            auto result = op.call(x, sh, sc);
            ctx->save_for_backward({x, sc});
            return result;
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, 
        torch::autograd::variable_list grad_output) {
			static auto op = torch::Dispatcher::singleton()
                .findSchemaOrThrow("w2kpops::modulate_backward_impl", "")
                .typed<decltype(modulate_backward_impl)>();
            auto saved_tensors = ctx -> get_saved_variables();
            auto grad_out = op.call(saved_tensors[0]/*x*/,saved_tensors[1]/*sc*/, grad_output[0]);
            return grad_out;
        }
    };

    torch::autograd::variable_list modulate_impl_autograd(const torch::Tensor& x, 
    const torch::Tensor& sh, const torch::Tensor& sc) {
        return Modulate::apply(x, sh, sc);
    }

    torch::Tensor modulate(const torch::Tensor& x, const torch::Tensor& sh, const torch::Tensor& sc) {
        static auto op = torch::Dispatcher::singleton()
            .findSchemaOrThrow("w2kpops::modulate_impl", "")
            .typed<decltype(modulate_forward_impl)>();
        auto result = op.call(x, sh, sc);
        return result[0];
    }
}