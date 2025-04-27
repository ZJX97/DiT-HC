import time
import torch


torch.ops.load_library("/pacific_ext/wxy/kpops_dir/build/lib/libkpprim.so")
# torch.ops.load_library("/pacific_ext/wxy/kpops_dir/lib/softmax_libkpprim.so")

shape = (28,256,3,16,72)

qkv=torch.rand(shape,dtype=torch.float32, requires_grad=True).permute(2,0,3,1,4)
sveqkv = qkv.data.clone().requires_grad_(True)
# flashqkv = qkv.data.clone().requires_grad_(True)

q,k,v = qkv.unbind(0)
sveq,svek,svev = sveqkv.unbind(0)
# flashq,flashk,flashv = flashqkv.unbind(0)


time1 = time.time()
qk = q@k.transpose(-2,-1)
x = qk.softmax(dim=-1)
ansori = x @ v
time2 = time.time()
print(f"ori forward time: {time2 - time1}")

time3 = time.time()
sveqk = sveq@svek.transpose(-2,-1)
y = torch.ops.wkpops.softmax(sveqk)
anssve = y @ svev
time4 = time.time()
print(f"sve forward time: {time4 - time3}")
print("forward check: ",torch.allclose(ansori,anssve,atol=0.00001))

qk.retain_grad()
sveqk.retain_grad()


time1 = time.time()
ansori.sum().backward()
time2 = time.time()
print(f"ori backward time: {time2 - time1}")


anssve.sum().backward()
time3 = time.time()
print(f"sve backward time: {time3 - time2}")


# print(qk.grad)
# print(sveqk.grad)
# z = torch.nn.functional.scaled_dot_product_attention(flashq,flashk,flashv,dropout_p = 0)
# time4 = time.time()
# print(f"flash time: {time4 - tim3}")
print("backward check: ", torch.allclose(qk.grad,sveqk.grad,atol=0.0001))
