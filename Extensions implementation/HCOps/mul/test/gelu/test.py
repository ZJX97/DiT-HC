import torch
# import kpops
import torch.nn.functional as F
import time
#shape = (4,4)
shape = (128,256,1152*4)

torch.ops.load_library("/pacific_fs/wxy/kpops_dir/build/lib/libkpprim.so")

torch.backends.mkldnn.enabled = False

A = (torch.rand(shape,dtype=torch.float32) * 2 - 1 ) * 1

B = F.gelu(A, approximate = "tanh")

torch_time = 0
sve_time = 0
iteration = 1

for i in range(iteration):   
    print(f"iteration: {i}")
    time1 = time.time()
    D = F.gelu(A,approximate = "tanh")
    time2 = time.time() - time1
    torch_time = torch_time + time2
    print(f"torch_time: {time2}")
    time1 = time.time()
    
    print(hex(A.data_ptr()))

    C = torch.ops.kpops.gelu(A)
    
    print(hex(C.data_ptr()))
    print(hex(id(C)))
    time2 = time.time() - time1
    sve_time = sve_time + time2
    print(f"sve_time: {time2}")
    print("is same: ",torch.allclose(C,D,atol=0.0001))

print("\n")
print(f"torch time:{torch_time / iteration}")
print(f"sve time:{sve_time / iteration}")
