#Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

#from ori_models import DiT_models
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)
from torch.distributed._tensor import Shard, Replicate
from torch.profiler import profile,record_function, ProfilerActivity 
from datasets import LatentDataset
from kpnn import kpLinear
import ProcessGroupHMPI
from kpext.mem.prefetcher import Prefetcher
#from torch.distributed import ProcessGroup
#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def replace_linear_with_kplinear(model, kplinear_class):
    #module_class_name = model.__class__.__name__.lower()
    #attention_keywords = ["attention", "attn", "multihead", "selfattention"]
    #is_attention_module = any(keyword in module_class_name for keyword in attention_keywords)

    #if is_attention_module:
    #    return model
    
    
    for name, module in list(model.named_children()):
        if isinstance(module, torch.nn.Linear):
            new_layer = kplinear_class(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None
            )
            if module.weight is not None:
                new_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None and new_layer.bias is not None:
                new_layer.bias.data.copy_(module.bias.data)
                            
            setattr(model, name, new_layer)
        else:
             replace_linear_with_kplinear(module, kplinear_class)
                                                                                          
    return model




@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                                  Training Loop                                #
#################################################################################
def train_loop(args,sampler,logger, loader,vae, diffusion, model, ema, opt, rank, world_size, checkpoint_dir, prof = None):
    # Variables for monitoring/logging purposes:

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        CNT  = 0
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            CNT += 1
            # x.shape: 3 X 256 X 256
            #print(f"rank={rank},vae embedding")
            if not args.pre_latent:
                with record_function("vae inference"):
                    with torch.no_grad():
    		        # Map input images to latent space + normalize latents:
                        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            # x.shape: 4 X 32 X 32
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],))
            

            model_kwargs = dict(y=y)
            # print(f"rank={rank}, forward")
            
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            

            # print(f"rank={rank}, loss.mean")
            loss = loss_dict["loss"].mean()
            # print(f"rank={rank}, zero grad")

            # opt.zero_grad()
            opt.zero_grad(set_to_none=False)
            

            # print(f"rank={rank}, backward")
            loss.backward()
#            with model.no_sync(): 
#                loss.backward()
#            #print(f"rank={rank}, opt step")
#            for param in model.parameters():
#                if param.grad is not None:
#                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)


            opt.step()
            #print(f"rank={rank}, update ema")

            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                secs_per_step = (end_time - start_time) / log_steps
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Sec/Steps:{secs_per_step:.2f}s, Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            if prof is not None:
                prof.step()
            if args.debug and CNT > 5:
                break 
            if args.demo and CNT > 5:
                break
        
        if args.debug:
            break
        if args.demo:
            break


def custom_collate(batch_samples):
    images = torch.cat([x for x,_ in batch_samples], dim = 0)
    labels = torch.cat([y for _,y in batch_samples], dim = 0)
    return images, labels

#################################################################################
#                                  Training control                                #
#################################################################################
def main(args):
    """
    Trains a new DiT model.
    """
    # Setup DDP:

    #patch_collectives() 
    #patch_ddp_directly()
    # dist.init_process_group("mpi")
    
    world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])

    dist.init_process_group("hmpi", world_size=world_size, rank=rank)

    torch.ops.load_library("/pacific_ext/wxy/kpops_dir/lib/linear-libkpprim-0331.so")
    torch.ops.load_library("/pacific_ext/wxy/kpops_dir/lib/wkpops-softmax_libkpprim.so")
    torch.ops.load_library("/pacific_ext/wxy/kpops_dir/lib/lkpops-attention_libkpprim-512X512.so")
    torch.ops.load_library("/pacific_ext/wxy/kpops_dir/lib/w2kpops-mul-libkpprim.so")
    
    #patch_collectives() 
#    torch.backends.mkldnn.enabled=True   
    # world_size = dist.get_world_size()
    # rank = dist.get_rank()

    tp_size = args.tp_size
    dp_size = world_size // tp_size

    if args.banOneDNN:
        torch.backends.mkldnn.enabled=False


    assert args.global_batch_size % dp_size == 0, f"Batch size must be divisible by dp size."

    dp_rank = rank // tp_size
    tp_rank = rank % tp_size

    seed = dp_rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank},tp_rank={tp_rank}, dp_rank={dp_rank}, seed={seed}, world_size={world_size}, dp_size={dp_size}, tp_size={tp_size}.")


    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        checkpoint_dir = None
        logger = create_logger(None)

    
    logger.info(f"INFO:: batch_size={args.global_batch_size},debug={args.debug},pre_latent={args.pre_latent}, world_size={world_size}, dp_size={dp_size}, tp_size={tp_size}.")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if args.debug:
        #model = profiler_models[args.model](
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        )
        
    else:
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        )

    model = replace_linear_with_kplinear(model, kpLinear)
    print("model init")

    custom_wrap_classes = {
        torch.nn.Sequential,
        torch.nn.ModuleList
    }
    def custom_auto_wrap_policy(module, recurse, **kwargs):
        if isinstance(module, tuple(custom_wrap_classes)):
            return recurse
        return True
    model = Prefetcher(model, auto_wrap_policy=custom_auto_wrap_policy, onload_sdma_cpu_offset = 36, offload_sdma_cpu_offset = 74)

    if tp_size > 1:
        mesh_2d = init_device_mesh("cpu",(dp_size,tp_size), mesh_dim_names = ("dp", "tp"))
        dp_mesh = mesh_2d["dp"]
        tp_mesh = mesh_2d["tp"]
        for block in model.blocks:
            layer_tp_plan = {
                # "adaLN_modulation.adaLN_Linear":RowwiseParallel(),
                "attn.qkv": ColwiseParallel(),
                "attn.proj": RowwiseParallel(),
                "mlp.fc1": ColwiseParallel(),
                "mlp.fc2": RowwiseParallel()
            }
            block.attn.num_heads = block.attn.num_heads // tp_size

            # block = parallelize_module(
            #     module=block,
            #     device_mesh=tp_mesh,
            #     parallelize_plan=PrepareModuleInput(
            #         input_layouts=(None,Replicate(),),
            #         desired_input_layouts=(None,Shard(0),),
            #     ),
            # )

            block = parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_tp_plan
            )
        model = DDP(model,device_mesh=dp_mesh)
    else:
        model = DDP(model, static_graph=True, gradient_as_bucket_view=True)
        # model = DDP(model)

    #disable_collective_ops()        
    print("DDP init")


    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model.module)  # Create an EMA of the model for use after training
    requires_grad(ema, False)


    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if not args.pre_latent: 
        vae = AutoencoderKL.from_pretrained(f"{args.pretrain_path}/pretrained/sd-vae-ft-{args.vae}")
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolder(args.data_path, transform=transform)
    else:
        vae = None
        dataset = LatentDataset(args.data_path)

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True, weight_decay=0)

    # Setup data:
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dp_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=1,
        collate_fn = custom_collate
)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    
    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    if args.debug:
        print("===============run with profiler================")

        with profile(
            activities=[
                ProfilerActivity.CPU,  # 监控 CPU 活动
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=4),
            profile_memory=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profiler_dir),
            record_shapes=False,  # 记录张量形状
            with_stack=False  # 记录调用堆栈
        ) as prof:
            train_loop(args,sampler,logger, loader,vae, diffusion, model, ema, opt, rank, world_size, checkpoint_dir, prof)
        if rank == 0:
            with open(f"{args.profiler_dir}/log", 'w', encoding='utf-8') as f:
                for sorted_key in ["cpu_time_total", "self_cpu_time_total", "cpu_memory_usage", "self_cpu_memory_usage"]:
                    f.write(f"==========================sorted by {sorted_key}==========================")
                    f.write(prof.key_averages().table(sort_by=sorted_key, row_limit=40))
    else:
        train_loop(args,sampler,logger, loader,vae, diffusion, model, ema, opt, rank, world_size, checkpoint_dir)

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/pacific_ext/zjx/train_res")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=4_000)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--pretrain-path", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--profiler-dir", type=str, default="./profiler_res/0407")
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--banOneDNN", action="store_true", default=False)
    parser.add_argument("--pre-latent", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
