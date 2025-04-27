import torch
#from ..src import kpops
import numpy as np
import kpops

A = torch.tensor([1, 2, 3], dtype=torch.float32)
B = torch.tensor([4, 5, 6], dtype=torch.float32)

C = torch.ops.kpops.vector_add(A, B)

print(C)
