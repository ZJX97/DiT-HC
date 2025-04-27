import os
import torch
import ProcessGroupHMPI
import torch.distributed as dist
import time
def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def all_reduce():
    #world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    #rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    #dist.init_process_group("hmpi", world_size=world_size, rank=rank)

    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    print("world_size:", world_size, "rank:", rank)
    #dist.init_process_group("hmpi")
    dist.init_process_group("hmpi", world_size=world_size, rank=rank)
    #dist.init_process_group("hmpi", init_method="tcp://localhost:29500", rank=rank, world_size=world_size)
    #exit(0)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    print("after init  world_size:", world_size, "rank:", rank)
    print("Type:", type(dist.distributed_c10d._get_default_group()))
    # print("Available Backends:", dist.Backend.__members__)
    # print("Available Backends:", dist.Backends.__members__)
    x = torch.ones(10000000)
    for _ in range(10):
        #print("sleep 20s & start allreduce")
        time.sleep(10)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        #time.sleep(20)

    print(x.mean())

if __name__ == "__main__":
    all_reduce()
