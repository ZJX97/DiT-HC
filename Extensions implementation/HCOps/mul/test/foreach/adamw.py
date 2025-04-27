import torch
# import kpops
import torch.nn.functional as F
import time

torch.ops.load_library("/pacific_fs/wxy/kpops_dir/build/lib/libkpprim.so")

def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    return x.item() if isinstance(x, torch.Tensor) else x


def torch_adamw(device_params,device_grads, device_exp_avgs, device_exp_avg_sqs, device_state_steps, lr, eps, beta1, beta2):
    torch._foreach_add_(
        device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
    ) 

    torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
    torch._foreach_mul_(device_exp_avg_sqs, beta2)
    torch._foreach_addcmul_(
        device_exp_avg_sqs, device_grads, device_grads, 1 - beta2
    )

    # Delete the local intermediate since it won't be used anymore to save on peak memory
    del device_grads

    bias_correction1: Union[Tuple[Tensor, ...], List[Tensor]]
    bias_correction2: Union[Tuple[Tensor, ...], List[Tensor]]
    bias_correction2_sqrt: Union[Tuple[Tensor, ...], List[Tensor]]

    bias_correction1 = [
        1 - beta1 ** _get_value(step) for step in device_state_steps
    ]
    bias_correction2 = [
        1 - beta2 ** _get_value(step) for step in device_state_steps
    ]

    step_size = [(lr / bc) * -1 for bc in bias_correction1]

    bias_correction2_sqrt = [
        bc**0.5 for bc in bias_correction2  # type: ignore[arg-type]
    ]
      
    exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)
    torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
    torch._foreach_add_(exp_avg_sq_sqrt, eps)
    torch._foreach_addcdiv_(
        device_params,
        device_exp_avgs,
        exp_avg_sq_sqrt,
        step_size,  # type: ignore[arg-type]
    )


if __name__ == '__main__':
    cnt = 1000
    sve_params = []
    torch_params = []

    sve_grads = []
    torch_grads = []

    sve_avgs = []
    torch_avgs = []

    sve_avg_sqs = []
    torch_avg_sqs = []

    sve_steps = []
    torch_steps = []

    for i in range(cnt):
        shape = (1152, 4, 2, 2)
        sve_param = torch.rand(shape,dtype=torch.float32)
        torch_param = sve_param.data.clone()
        sve_params.append(sve_param)
        torch_params.append(torch_param)

        sve_grad = torch.rand(shape,dtype=torch.float32)
        torch_grad = sve_grad.data.clone()
        sve_grads.append(sve_grad)
        torch_grads.append(torch_grad)

        sve_avg = torch.rand(shape,dtype=torch.float32)
        torch_avg = sve_avg.data.clone()
        sve_avgs.append(sve_avg)
        torch_avgs.append(torch_avg)

        sve_avg_sq = torch.rand(shape,dtype=torch.float32)
        torch_avg_sq = sve_avg_sq.data.clone()
        sve_avg_sqs.append(sve_avg_sq)
        torch_avg_sqs.append(torch_avg_sq)
        
        sve_steps.append(torch.tensor(1.0))
        torch_steps.append(torch.tensor(1.0))


    lr = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    time1 = time.time()

    torch_adamw(torch_params, torch_grads, torch_avgs, torch_avg_sqs, torch_steps, lr, eps, beta1, beta2)

    time2 = time.time()

    torch.ops.kpops.foreach_adamw(sve_params, sve_grads, sve_avgs, sve_avg_sqs, sve_steps, beta1, beta2,eps,lr)

    time3 = time.time()

    # for i in range(cnt):
    #     print("params: ", torch.allclose(sve_params[i],torch_params[i],atol=0.00000001))
    #     print("grads: ", torch.allclose(sve_grads[i],torch_grads[i],atol=0.00000001))
    #     print("avgs: ",torch.allclose(sve_avgs[i],torch_avgs[i],atol=0.00000001))
    #     print("avg_sqs: ",torch.allclose(sve_avg_sqs[i],torch_avg_sqs[i],atol=0.00000001))
    #     print("steps: ",torch.allclose(sve_steps[i],torch_steps[i],atol=0.00000001))
    print(sve_steps[0])
    print(sve_params[0])
    print(sve_grads[0])
    print(sve_avgs[0])
    print(sve_avg_sqs[0])
    print(f"torch time: {time2 - time1}")
    print(f"sve time: {time3 - time2}")