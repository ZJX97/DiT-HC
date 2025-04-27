#include "../ops.hpp"
#include "../torch_impl.hpp"
#include<cstdio>
#include<iostream>
#include<cstdlib>
#include<time.h>

namespace kpops::torch_impl {
    torch::autograd::variable_list mla_forward_impl(const torch::Tensor& x, 
        const torch::Tensor& gate, const torch::Tensor& y) {
            torch::Tensor result = torch::empty_like(x);
            kpops::mlaKernel(x, gate, y, result);
		return {result};
    }
    torch::autograd::variable_list mla_backward_impl(const torch::Tensor& gate,
        const torch::Tensor& y, const torch::Tensor& grad_output) {
		
        torch::Tensor grad_gate = torch::zeros_like(gate);
        torch::Tensor grad_y = torch::empty_like(y);

        assert(grad_output.stride(2) == 1 && "mla backward: grad_output.stride(2) should be 1");

        kpops::mlaBackwardKernel(gate, y, grad_output, grad_gate, grad_y);

        // std::cout<<"grad stride: " << grad_output.stride(0)<<" "<<grad_output.stride(1)<<" "<<grad_output.stride(2)<<std::endl;
        // std::cout<<"grad shape: " << grad_output.size(0)<<" "<<grad_output.size(1)<<" "<<grad_output.size(2)<<std::endl;
        // kpops::mlaBackwardKernel(x ,sc, grad_x,grad_sh, grad_sc, grad_output);

        return {grad_output, grad_gate, grad_y};
    }
	
	class Mla : public torch::autograd::Function<Mla> {
    public:
        static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, 
        const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& y) {
            at::AutoDispatchBelowADInplaceOrView guard;
			static auto op = torch::Dispatcher::singleton()
                .findSchemaOrThrow("w2kpops::mla_impl", "")
                .typed<decltype(mla_forward_impl)>();
            auto result = op.call(x, gate, y);
            ctx->save_for_backward({gate,y});
            return result;
        }

        static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, 
        torch::autograd::variable_list grad_output) {
			static auto op = torch::Dispatcher::singleton()
                .findSchemaOrThrow("w2kpops::mla_backward_impl", "")
                .typed<decltype(mla_backward_impl)>();
            auto saved_tensors = ctx -> get_saved_variables();
            auto grad_out = op.call(saved_tensors[0]/*gate*/,saved_tensors[1]/*y*/, grad_output[0]);
            return grad_out;
        }
    };

    torch::autograd::variable_list mla_impl_autograd(const torch::Tensor& x, 
    const torch::Tensor& gate, const torch::Tensor& y) {
        return Mla::apply(x, gate, y);
    }

    torch::Tensor mla(const torch::Tensor& x, const torch::Tensor& gate, const torch::Tensor& y) {
        static auto op = torch::Dispatcher::singleton()
            .findSchemaOrThrow("w2kpops::mla_impl", "")
            .typed<decltype(mla_forward_impl)>();
        auto result = op.call(x, gate, y);
        return result[0];
    }
}