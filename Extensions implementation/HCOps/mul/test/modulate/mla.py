import torch
import torch.nn as nn
import time
torch.ops.load_library("/pacific_ext/wxy/kpops_mul/build/lib/libkpprim.so")

N = 16
T = 256
D = 1152
tx = torch.randn(N, T, D)
x = torch.as_strided(
    tx,
    size=(N,T,D),
    stride=(T*D,1,T)
)
x.requires_grad_(True)

adaLN = torch.randn(N,D*6, requires_grad=True)
_,_,gate,_,_,_ = adaLN.chunk(6,dim = 1)
gate.retain_grad()
adaLN.retain_grad()

y = torch.randn(N,T,D, requires_grad=True)
y.retain_grad()
x = x + gate.unsqueeze(1) * y

grad = torch.ones_like(x)
x.backward(grad)


svetx = tx.data.clone()
svex = torch.as_strided(
    svetx,
    size=(N,T,D),
    stride=(T*D,1,T)
)
svex.requires_grad_(True)

sveadaLN = adaLN.data.clone()
sveadaLN.requires_grad_(True)
sveadaLN.retain_grad()

_,_,svegate,_,_,_ = sveadaLN.chunk(6,dim = 1)
svegate.retain_grad()
svey = y.data.clone()

svey.requires_grad_(True)
svey.retain_grad()
svex = torch.ops.w2kpops.mla(svex, svegate, svey)

grad2 = torch.ones_like(svex)
svex.backward(grad2)


# print(gate.grad)
# print(svegate.grad)
print(torch.allclose(gate.grad,svegate.grad,atol=0.0001))
