import torch


x = torch.rand(128,256,1024)
x = x.transpose(-1,-2)
y = torch.empty_like(x)
print(x.stride())
print(y.stride())
