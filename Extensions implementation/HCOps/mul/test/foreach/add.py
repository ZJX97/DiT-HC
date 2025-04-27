import torch
# import kpops
import torch.nn.functional as F
import time

torch.ops.load_library("/pacific_fs/wxy/kpops_dir/build/lib/libkpprim.so")
shape = (128,256,1152*4)

A = torch.rand(shape,dtype=torch.float32) * 2 - 1

time1 = time.time()
torch.ops.kpops.foreach_add_impl([A],  1, 1)
time2 = time.time()
torch._foreach_add_([A], torch.tensor(1.0,device="cpu"), alpha = 1.0)
time3 = time.time()

print(f"sve time: {time2 - time1}")
print(f"torch time: {time3 - time2}")