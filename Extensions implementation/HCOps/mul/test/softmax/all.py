import time
import torch
import torch.nn as nn
torch.backends.mkldnn.enabled = True

# torch.ops.load_library("/pacific_ext/wxy/kpops_dir/lib/softmax_libkpprim.so")
# torch.ops.load_library("/pacific_fs/wxy/kpops_dir/lib/softmax_libkpprim.so")
B = 28
N = 256
C = 1152

x = torch.rand(B,N,C)

wqkv = nn.Linear(C,C*3)

qkv = wqkv(x).reshape(B,N,3,16,72).permute(2,0,3,1,4)
q,k,v = qkv.unbind(0)
qk = q@k.transpose(-2,-1)
x = qk.softmax(dim=-1)
ansori = x @ v
ansori.sum().backward()



