from pathlib import Path
import torch
import glob
import itertools
from ..utils.logging import print0

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int, cfg, print_stats=True):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise ValueError(f"No files found matching pattern: {filename_pattern}")
    
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    
    # Calculate total tokens across all shards
    total_tokens = 0
    tokens_per_file = []
    for file in files:
        header = torch.from_file(str(file), False, 256, dtype=torch.int32)
        file_tokens = int(header[2])
        total_tokens += file_tokens
        tokens_per_file.append(file_tokens)
    
    # Calculate how many tokens we need for training
    tokens_needed = args.train_steps * batch_size * args.grad_acc_steps
    
    # Determine if we need to cycle and calculate epochs
    will_cycle = total_tokens < tokens_needed
    epochs = tokens_needed / total_tokens if total_tokens > 0 else 0
    
    if rank == 0 and print_stats:
        print0(f"Total tokens across {len(files)} shard(s): {total_tokens:,}", console=True)
        print0(f"Tokens needed for {args.train_steps} iterations: {tokens_needed:,}", console=True)
        print0(f"Training will use approximately {epochs:.2f} epochs over the data", console=True)
    
    file_iter = itertools.cycle(files) if will_cycle else iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets