import time
import torch


torch.ops.load_library("/pacific_ext/wxy/kpops_dir/build/lib/libkpprim.so")
# torch.ops.load_library("/pacific_fs/wxy/kpops_dir/lib/softmax_libkpprim.so")

shape = (28,256,3,16,72)

qkv=torch.rand(shape,dtype=torch.float32, requires_grad=True).permute(2,0,3,1,4)
sveqkv = qkv.data.clone().requires_grad_(True)
flashqkv = qkv.data.clone().requires_grad_(True)

q,k,v = qkv.unbind(0)
sveq,svek,svev = sveqkv.unbind(0)
flashq,flashk,flashv = flashqkv.unbind(0)


qk = q@k.transpose(-2,-1)
sveqk = sveq@svek.transpose(-2,-1)

time1 = time.time()
x = qk.softmax(dim=-1)

time2 = time.time()

print(f"ori time: {time2 - time1}")

y = torch.ops.kpops.softmax(sveqk)
time3 = time.time()
print(f"sve time: {time3 - time2}")

# z = torch.nn.functional.scaled_dot_product_attention(flashq,flashk,flashv,dropout_p = 0)
# time4 = time.time()
# print(f"flash time: {time4 - tim3}")

print(torch.allclose(x,y,atol=0.00001))
