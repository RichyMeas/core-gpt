import os
import sys
import time
import csv
import random
import numpy as np # Import numpy for potential future use, set random seed now not to forget to set it later

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Set to the GPU you want to use, or leave it as is for all GPUs
os.environ["TORCH_DYNAMO_DISABLE_INDUCTOR"] = "1"
#os.environ["TORCH_LOGS"] = "+dynamo"
#os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
print(torch.__version__)
print(torch.cuda.is_available())
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
#torch._inductor.config.max_autotune_gemm_backends = ["ATEN"]

import omegaconf
import hydra
from omegaconf import DictConfig, OmegaConf

from ..models.gpt_config import Hyperparameters
from ..models.gpt_model import GPT
from ..utils.log_utils import print0, initalize_logging
from ..utils.kernel_warmup import warmup_kernels
from ..data_loaders.text import distributed_data_generator
from ..optimizers.muon import Muon
from ..optimizers.utils import get_lr

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def initialize_model(cfg, world_size, master_process, logfile):
    model: nn.Module = GPT(vocab_size=cfg.vocab_size, 
        num_layers=cfg.num_layers,
        num_val_emb=cfg.num_val_emb,
        num_heads=cfg.num_heads, 
        model_dim=cfg.model_dim,
        max_seq_len=max(cfg.data.train_seq_len, cfg.data.val_seq_len),
        mlp_ratio=cfg.mlp_ratio,
        head_dim=cfg.head_dim,
    ).cuda()
    print0(f'{model.get_num_params()} parameters', master_process=master_process, logfile=logfile, console=True)
    print0(model, master_process=master_process, logfile=logfile)

    # Set FP8 option based on hyperparameters
    model.lm_head.use_fp8 = cfg.use_fp8

    for m in model.modules():
        #if isinstance(m, nn.Embedding):
        m.bfloat16()
    if world_size > 1:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
    return model

def intialize_optimizers(model, world_size, rank):
    # collect the parameters to optimize
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # init the optimizer(s)
    adam_params = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)

    # For single GPU case, we need to modify how Muon is initialized
    if world_size == 1:
        # Create update buffer for single GPU
        for param in hidden_matrix_params:
            param.requires_grad_(True)
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)
    else:
        optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, rank=rank, world_size=world_size)

    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers

@hydra.main(version_base=None, config_path="./configs", config_name="train_gpt.yaml")
def main(cfg):
    OmegaConf.resolve(cfg)
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Check if environment variables are set by torchrun, otherwise default to single GPU
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        # Multi-GPU setup with torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single GPU setup
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

    print(f"Running with {world_size} GPU{'s' if world_size > 1 else ''}")
    assert torch.cuda.is_available()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Initialize distributed process group if using multiple GPUs
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = (rank == 0)  # this process will do logging, checkpointing etc.

    logfile, experiment_dir_path, metrics_csv_path, tb_writer = initalize_logging(master_process, cfg)

    # log information about the hardware/software environment this is running on
    print0(f"Running Python {sys.version}", master_process=master_process, logfile=logfile)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}", master_process=master_process, logfile=logfile)
    def nvidia_smi():
        import subprocess  # avoid top level import
        return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
    print0(nvidia_smi(), master_process=master_process, logfile=logfile)
    print0("="*100, master_process=master_process, logfile=logfile)

    #################################################
    #########      Seed for Reproducibility     #####
    #################################################

    # Set the seed *before* initializing the model or optimizer
    if cfg.seed is not None:
        print0(f"Setting random seed to {cfg.seed} for model initialization", master_process=master_process, logfile=logfile, console=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed) # Important for multi-GPU consistency
            # The following might be needed for full determinism, but can impact performance
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    ########################################
    #    Construct model and optimizer     #
    ########################################

    model: nn.Module = initialize_model(cfg, world_size, master_process=master_process, logfile=logfile)

    optimizers = intialize_optimizers(model, world_size, rank)
    optimizer2 = optimizers[1] # Muon optimizer

    # Use a more memory-efficient compilation option
    if cfg.use_fp8:
        model: nn.Module = torch.compile(model, dynamic=False)
    else:
        model: nn.Module = torch.compile(model, dynamic=False)

    # Add fallback mode to handle compilation errors
    #import torch._dynamo

    ########################################
    #            Warmup kernels            #
    ########################################

    print0("warming up kernels...", master_process=master_process, logfile=logfile, console=True)

    # Attempt to limit memory fragmentation
    if hasattr(torch.cuda, 'memory_stats'):
        print0(f"Initial GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB", master_process=master_process, logfile=logfile)

    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    warmup_kernels(model, optimizers, cfg, world_size)

    if hasattr(torch.cuda, 'memory_stats'):
        print0(f"After warmup GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB", master_process=master_process, logfile=logfile)

    print0("kernels are toasty", master_process=master_process, logfile=logfile, console=True)

    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(cfg.data.train_files, world_size * cfg.data.train_seq_len, rank, world_size, cfg, master_process=master_process, logfile=logfile)

    training_time_ms = 0
    # start the clock
    if world_size > 1:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    token_count = 0
    for step in range(cfg.train_steps + 1):
        last_step = (step == cfg.train_steps)

        if step != 0 and not last_step and (step % cfg.checkpoint_interval == 0):
            # Save model checkpoint inside the experiment directory
            if master_process and cfg.get("save_model", False) and experiment_dir_path:
                log = dict(step=step, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                # Ensure experiment_dir_path exists (though it should from earlier)
                os.makedirs(experiment_dir_path, exist_ok=True)
                save_path = experiment_dir_path / f"state_step{step:06d}.pt"
                torch.save(log, str(save_path))
                print0(f"Saved checkpoint to {save_path}", master_process=master_process, logfile=logfile, console=True)

        # --------------- VALIDATION SECTION -----------------
        if step != 0 and (last_step or (cfg.val_loss_every > 0 and step % cfg.val_loss_every == 0)):
            # stop the clock
            if world_size > 1:
                torch.cuda.synchronize()
            # Note: training_time_ms accumulates *only* the time spent in the training loop
            # It does not include time spent in validation or other operations outside the loop
            training_time_ms += 1000 * (time.perf_counter() - t0)
            
            model.eval()
            
            # Ensure we validate on enough tokens while keeping memory usage reasonable
            val_batch_size = world_size * cfg.data.val_seq_len
            val_tokens_used = val_batch_size * cfg.val_steps
            print0(f"Validating on {val_tokens_used} tokens ({cfg.val_steps} steps with {val_batch_size} batch size)", master_process=master_process, logfile=logfile, console=True)
            
            val_loader = distributed_data_generator(cfg.data.val_files, val_batch_size, rank, world_size, cfg, master_process=master_process, logfile=logfile, print_stats=False)
            val_loss = 0
            with torch.no_grad():
                for i in range(cfg.val_steps):
                    inputs, targets = next(val_loader)
                    # Check if inputs exceed sequence length
                    if inputs.size(0) > cfg.data.val_seq_len:
                        inputs = inputs[:cfg.data.val_seq_len]
                        targets = targets[:cfg.data.val_seq_len]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        val_loss += model(inputs, targets)
            val_loss /= cfg.val_steps

            if tb_writer:
                tb_writer.add_scalar("val_loss", val_loss.item(), step) # Log validation loss to TensorBoard

            del val_loader
            if world_size > 1:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            
            # Calculate average time per step up to this point
            step_avg_ms = training_time_ms / max(step, 1) 
            print0(f"step:{step}/{cfg.train_steps} val_loss:{val_loss:.4f} "
                    f"train_time:{training_time_ms:.0f}ms step_avg:{step_avg_ms:.2f}ms", master_process=master_process, logfile=logfile, console=True)
            
            # Log validation metrics to CSV
            if master_process and metrics_csv_path:
                with open(metrics_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Use .item() to get float from tensor for val_loss
                    writer.writerow([step, 
                        "val", f"{val_loss.item():.4f}", 
                        f"{training_time_ms:.0f}", 
                        f"{step_avg_ms:.2f}"])

            if last_step: # inside validation section to avoid the if check every training iteration
                # 5. Save model checkpoint inside the experiment directory
                if master_process and cfg.get("save_model", False) and experiment_dir_path:
                    log = dict(step=step, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                    # Ensure experiment_dir_path exists (though it should from earlier)
                    os.makedirs(experiment_dir_path, exist_ok=True)
                    save_path = experiment_dir_path / f"state_step{step:06d}.pt"
                    torch.save(log, str(save_path))
                    print0(f"Saved checkpoint to {save_path}", master_process=master_process, logfile=logfile, console=True)
                # the last step only has the validation loop, so break to avoid training
                break
            
            model.train()
            # start the clock again for the next training segment
            if world_size > 1:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

        # --------------- TRAINING SECTION -----------------
        loss = torch.tensor([0.], device="cuda")
        for _ in range(cfg.grad_acc_steps):
            inputs, targets = next(train_loader)
            # Check if inputs exceed sequence length - can happen if the dataset has different sized examples
            if inputs.size(0) > cfg.data.train_seq_len:
                inputs = inputs[:cfg.data.train_seq_len]
                targets = targets[:cfg.data.train_seq_len]
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Forward pass
                step_loss = model(inputs, targets)
            step_loss = model(inputs, targets)
            loss += step_loss / cfg.grad_acc_steps
            token_count += inputs.numel() # Count tokens processed in this step

        if tb_writer:
            tb_writer.add_scalar("train_loss", loss.item(), step) # Log training loss to TensorBoard
            tb_writer.add_scalar("token_count", token_count, step) # Log training loss per step to TensorBoard

        loss.backward()

        if tb_writer:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            tb_writer.add_scalar("grad_norm", total_norm, step)

        if world_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step, cfg.train_steps, cfg.cooldown_frac)
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers
        for opt in optimizers:
            opt.step()

        if tb_writer:
            for i, opt in enumerate(optimizers):
                for j, group in enumerate(opt.param_groups):
                    tb_writer.add_scalar(f"lr/optimizer{i}_group{j}", group["lr"], step)

        # null the gradients
        model.zero_grad(set_to_none=True)
            
        # calculate *approximate* cumulative time and step average for logging during training
        # Note: This is approximate because it includes the time for the current step's forward/backward pass
        # The more precise time is recorded just before validation
        if master_process:
            if world_size > 1:
                torch.cuda.synchronize() # Ensure accurate timing up to this point
            # Calculate time elapsed since the end of the last validation phase
            current_segment_duration_ms = 1000 * (time.perf_counter() - t0) 
            # Calculate the *true* approximate cumulative time
            approx_cumulative_time_ms = training_time_ms + current_segment_duration_ms
            approx_step_avg_ms = approx_cumulative_time_ms / (step + 1)
            print0(f"step:{step+1}/{cfg.train_steps} "
                    f"train_time:{approx_cumulative_time_ms:.0f}ms "
                    f"step_avg:{approx_step_avg_ms:.2f}ms", master_process=master_process, logfile=logfile, console=True)
            if tb_writer:
                tb_writer.add_scalar("train_time", approx_cumulative_time_ms, step + 1)
                tb_writer.add_scalar("step_avg_time", approx_step_avg_ms, step + 1)
            # Log training step timing to CSV
            if metrics_csv_path:
                with open(metrics_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Loss is not typically calculated per training step here, add loss logging if needed
                    writer.writerow([step + 1, "train", f"{loss.item():.4f}", f"{approx_cumulative_time_ms:.0f}", f"{approx_step_avg_ms:.2f}"])

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", master_process=master_process, logfile=logfile, console=True)

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()