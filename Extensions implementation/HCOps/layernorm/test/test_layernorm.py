import torch
import torch.nn as nn
import numpy as np
#from ..src import kpops
import kpops

def test_forward_cpu(shape=(16, 256, 1152), eps=1e-5, affine=True):
    """仅测试 CPU 前向计算的数值一致性"""
    # 固定随机种子保证可重复性
    with torch.no_grad():  # 禁止梯度更新
        np.random.seed(42)
        torch.manual_seed(42)
        
        # 生成输入数据 (CPU + float32)
        x_np = np.random.randn(*shape).astype(np.float32)
        x = torch.from_numpy(x_np).float()  # 明确使用 float32
        print(x.shape)
        y_custom = torch.zeros_like(x)
        y_native = torch.zeros_like(x)

        M = shape[0] * shape[1]
        N = shape[-1]
        # 初始化参数
        if affine:
            weight = torch.randn(shape[-1])
            bias = torch.randn(shape[-1])
            mean = torch.zeros(M)
            rstd = torch.zeros(M)
        else:
            weight, bias = None, None
        #C = torch.ops.kpops.vector_add(x, y_custom)
        #print(C) 
        # 自定义实现前向
        torch.ops.kpops.kp_layer_norm(x, shape[-1:], eps, y_custom, mean, rstd, weight, bias)
        print("sucess custom forward from python")    
        # 原生实现前向
        native_ln = nn.LayerNorm(normalized_shape=shape[-1:], eps=eps, elementwise_affine=affine)
        if affine:
            with torch.no_grad():  # 禁止梯度更新
                native_ln.weight.copy_(weight)
                native_ln.bias.copy_(bias)
        y_native = native_ln(x)
        
        # 数值对比
        diff = (y_custom - y_native).abs()
        print(f"Shape: {shape}, Affine: {affine}")
        print(f"  Max diff: {diff.max().item():.2e}")
        print(f"  Mean diff: {diff.mean().item():.2e}")
        
        ## 严格断言 (容忍度可根据需求调整)
        #torch.testing.assert_allclose(y_custom, y_native, rtol=1e-5, atol=1e-6)
        #print("Test passed!\n")

if __name__ == "__main__":
    # 加载自定义算子 (替换为你的实际路径)
    #torch.ops.load_library("./my_ops.so")
    
    # 测试不同配置
    test_configs = [
        {"shape": (16, 256, 1152), "eps": 1e-5, "affine": True},
    ]
    
    for config in test_configs:
        test_forward_cpu(**config)
