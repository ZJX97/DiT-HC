import torch
import torch.nn as nn
import time
torch.ops.load_library("/pacific_ext/wxy/kpops_mul/build/lib/libkpprim.so")

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


N=16
T=256
D=1152

adaLN = nn.Linear(D,6*D)
# nn.init.xavier_normal_(adaLN.weight)

# sveadaLN = nn.Linear(D,6 * D)
# sveadaLN.weight.data.copy_(adaLN.weight.data)
# nn.init.zeros_(sveadaLN.bias)

# print(adaLN.weight)
# print(sveadaLN.weight)

svex = torch.rand(N,T,D, requires_grad=True)
x = svex.data.clone().requires_grad_(True)
svec = torch.rand(N,D, requires_grad=True)
c = svec.data.clone().requires_grad_(True)

c.retain_grad()
svec.retain_grad()
x.retain_grad()
svex.retain_grad()

shift_msa,scale_msa,_,_,_,_ = adaLN(c).chunk(6,dim = 1)
shift_msa.retain_grad()
scale_msa.retain_grad()
grad_output = torch.ones_like(x)
time1 = time.time()
y = modulate(x,shift_msa, scale_msa)
y.backward(grad_output)
time2 = time.time()


shift_msa,scale_msa,_,_,_,_ = adaLN(c).chunk(6,dim = 1)
shift_msa.retain_grad()
scale_msa.retain_grad()
grad_output = torch.ones_like(x)
time1 = time.time()
y = modulate(x,shift_msa, scale_msa)
y.backward(grad_output)
time2 = time.time()

print(f"ori time: {time2 - time1}")


sve_shift_msa,sve_scale_msa,_,_,_,_ = adaLN(svec).chunk(6,dim=1)
sve_shift_msa.retain_grad()
sve_scale_msa.retain_grad()
svegrad_output = torch.ones_like(svex)
time1 = time.time()
svey = torch.ops.wkpops.modulate(svex, sve_shift_msa, sve_scale_msa)
svey.backward(svegrad_output)
time2 = time.time()
print(f"sve time: {time2 - time1}")

# print(shift_msa.grad)
# print(c.grad)
# print(svec.grad)



print(torch.allclose(y,svey, atol=0.0001))
print(torch.allclose(c.grad,svec.grad,atol=0.0001))
