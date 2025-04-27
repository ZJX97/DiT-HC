import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
#import kpops

class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        #if input.dim() != 2 or weight.dim() != 2:
        #    raise ValueError("CustomLinear expects 2D input and weight tensors")
        #    
        #if input.dtype != torch.float32 or weight.dtype != torch.float32:
        #    raise TypeError("CustomLinear expects float32 tensors")
        #    
        input_shape = input.shape
        input_flatten = input.view(-1,input.shape[-1])
        ctx.save_for_backward(input_flatten, weight, bias)
        ctx.input_shape = input_shape 
        batch_size, in_features = input_flatten.size()
        out_features = weight.size(0)
        output = torch.zeros(batch_size, out_features, dtype=torch.float32, device=input.device)
        
        if bias is None:
            bias = torch.zeros(out_features, dtype=torch.float32, device=input.device)
        
        torch.ops.kpops.kp_linear_forward(input_flatten, weight, bias, output)
        output = output.view(*input_shape[:-1], weight.size(0)) 
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_output_flat = grad_output.view(-1, grad_output.shape[-1]) 
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        
        torch.ops.kpops.kp_linear_backward(
            grad_output_flat, input, weight, grad_input, grad_weight, grad_bias
        )
        grad_input = grad_input.view(*(ctx.input_shape)) 
        if bias is None:
            return grad_input, grad_weight, None
        else:
            return grad_input, grad_weight, grad_bias


class kpLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(kpLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        #print("input size:", input.size(), input.shape)
        #print("weight size:", self.weight.size())
        #print("bias size:", self.bias.size())
        return CustomLinearFunction.apply(input, self.weight, self.bias)
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


# 使用示例
import math

def test_custom_linear():
    # 设置随机种子以便结果可重现
    torch.manual_seed(42)
    
    # 创建输入和模型
    batch_size = 7168
    in_features = 1152
    out_features = 4608
    
    # 创建输入
    x = torch.randn(batch_size, in_features, dtype=torch.float32)
    
    # 创建自定义线性层和 PyTorch 线性层
    custom_linear = kpLinear(in_features, out_features)
    torch_linear = nn.Linear(in_features, out_features)
    
    # 确保两个模型具有相同的参数
    with torch.no_grad():
        torch_linear.weight.copy_(custom_linear.weight)
        torch_linear.bias.copy_(custom_linear.bias)
    
    # 前向传播
    custom_output = custom_linear(x)
    torch_output = torch_linear(x)
    
    # 比较结果
    error = torch.abs(custom_output - torch_output).max().item()
    print(f"Maximum forward error: {error:.9f}")
    
    # 创建梯度并进行反向传播
    grad = torch.randn_like(custom_output)
    custom_output.backward(grad)
    torch_output.backward(grad)
    
    # 比较梯度
    weight_error = torch.abs(custom_linear.weight.grad - torch_linear.weight.grad).max().item()
    bias_error = torch.abs(custom_linear.bias.grad - torch_linear.bias.grad).max().item()
    print(f"Maximum weight gradient error: {weight_error:.9f}")
    print(f"Maximum bias gradient error: {bias_error:.9f}")

if __name__ == "__main__":
    # 请确保你已经加载了自定义算子库后再运行此测试
    # 取消下面一行的注释来运行测试
    test_custom_linear()
