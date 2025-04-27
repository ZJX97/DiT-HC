import torch
import time
import torch.nn.functional as F
import kpops
import os
# torch.backends.mkldnn.enabled = False


def calc(my_tensor, torch_tensor, dnn_tensor):
    torch.backends.mkldnn.enabled = False

    time1 = time.time()
    my_gelu = torch.ops.kpops.gelu(my_tensor)
    time2 = time.time()
    torch_gelu = F.gelu(torch_tensor, approximate="tanh")
    time3 = time.time()

    time4 = time.time()
    my_loss = my_gelu.sum()
    my_loss.backward()
    time5 = time.time()
    torch_loss = torch_gelu.sum()
    torch_loss.backward()
    time6 = time.time()

    torch.backends.mkldnn.enabled = True

    time7 = time.time()
    dnn_gelu = F.gelu(dnn_tensor, approximate="tanh")
    time8 = time.time()
    dnn_loss = dnn_gelu.sum()
    dnn_loss.backward()
    time9 = time.time()

    # print(torch.allclose(my_tensor.grad,torch_tensor.grad,atol=0.00000001))

    # sve_for,torch_for,dnn_for, sve_back,torch_back,dnn_back
    return time2 - time1,time3 - time2,time8 - time7, time5 - time4, time6 - time5,time9 - time8

def avg_calc(threads_num, shape):
    t_sve_for = 0
    t_sve_back = 0
    t_torch_for = 0
    t_torch_back = 0
    t_dnn_for = 0
    t_dnn_back = 0

    torch.set_num_threads(threads_num)
    print(f"OMP_NUMS: {threads_num}")

    it = 10

    for i in range(it):
        my_tensor = torch.rand(shape, requires_grad=True)
        torch_tensor = my_tensor.data.clone().requires_grad_(True)
        dnn_tensor = my_tensor.data.clone().requires_grad_(True)

        sve_for,torch_for,dnn_for,sve_back,torch_back,dnn_back = calc(my_tensor,torch_tensor,dnn_tensor)
        
        t_sve_for += sve_for
        t_sve_back += sve_back
        t_torch_for += torch_for
        t_torch_back += torch_back
        t_dnn_for += dnn_for
        t_dnn_back += dnn_back
        
    return t_sve_for / it, t_torch_for / it,t_dnn_for / it, t_sve_back / it, t_torch_back / it, t_dnn_back / it

if __name__ == '__main__':

    torch.manual_seed(54) 
    shape = (128,256,1152*4)
    torch_forward_time = []
    torch_backward_time = []
    sve_forward_time = []
    sve_backward_time = []    
    dnn_forward_time = []
    dnn_backward_time = []

    # threads = [1,2,4,6,8,10,15,20,30,50,100,150]
    threads = [100]

    for i in threads:
        sve_for,torch_for,dnn_for,sve_back,torch_back,dnn_back = avg_calc(i, shape)
        sve_forward_time.append(sve_for)
        sve_backward_time.append(sve_back)
        torch_forward_time.append(torch_for)
        torch_backward_time.append(torch_back)
        dnn_forward_time.append(dnn_for)
        dnn_backward_time.append(dnn_back)


    with open("data.txt","w") as f:
        f.write("======sve forward time=====\n")
        f.write(str(sve_forward_time) + "\n")
        f.write("======torch forward time=====\n")
        f.write(str(torch_forward_time) + "\n")
        f.write("======dnn forward time=====\n")
        f.write(str(dnn_forward_time) + "\n")
        f.write("======sve backward time=====\n")
        f.write(str(sve_backward_time) + "\n")
        f.write("======torch backward time=====\n")
        f.write(str(torch_backward_time) + "\n")
        f.write("======dnn backward time=====\n")
        f.write(str(dnn_backward_time) + "\n")
    
# print(torch.min(my_tensor), torch.max(my_tensor))

# print(my_tensor)
# print(torch_tensor)


# print(f"sve forward: {time2 - time1}s")
# print(f"torch forward: {time3 - time2}s")
# print("forward is same: ",torch.allclose(my_gelu,torch_gelu,atol=0.000000001))


# print(my_gelu.grad)
# print(torch_gelu.grad)

# print(my_tensor.grad)
# print(torch_tensor.grad)

# print(f"sve backward: {time5 - time4}s")
# print(f"torch backward: {time6 - time5}s")
# print("backward is same: ",torch.allclose(my_tensor.grad,torch_tensor.grad,atol=0.000000001))
