from dataclasses import dataclass
import argparse
from typing import Optional

@dataclass
class Hyperparameters:
    """
    default values are set to fit on a 2x GPUs w/ 8GB of VRAM each, but are not necessarily optimal
    """
    model_name: str = "ModdedGPT"
    # data
    train_files: str = "data/fineweb*_train_*.bin" # input .bin to train on
    val_files: str = "data/fineweb*_val_*.bin" # input .bin to eval validation loss on
    train_seq_len: int = 8 * 1024 # FlexAttention sequence length
    val_seq_len: int = 16 * 1024 # FlexAttention sequence length for validation (should be able to fit more than train_seq_len)
    # optimization loop
    val_steps: int = 10 # number of steps to run validation for
    train_steps: int = 20 #_000 # number of training steps to run
    grad_acc_steps: int = 1 # number of gradient accumulation steps per training step
    cooldown_frac: int = 0.4 # fraction of training spent cooling down the learning rate
    # architecture
    tokenizer: str = "gpt4regex_v50256_n1000000000.pkl" # any .pkl file in tokenizers/
    vocab_size: int = 50257 # should be the tokenizer's size plus any special tokens
    # model size - parameters set for GPUs w/ 8GB VRAM
    num_layers: int = 12  # number of reansformer blocks
    num_heads: int = 6   # number of attention heads
    model_dim: int = 384  # size of model embedding vectors
    head_dim: Optional[int] = None  # size of attention heads; if None, will default to model_dim // num_heads
    mlp_ratio: int = 4  # MLP hidden dimension is model_dim * mlp_ratio
    num_val_emb: int = 2 # number of value embeddings used at initial and final layers
    # memory optimization 
    use_fp8: bool = False # experimental; True on H100s (and newer?) should improve performance but seems to use more vram somehow
    # evaluation and logging
    val_loss_every: int = 100 # every how many steps to evaluate val loss? 0 for only at the end
    save_model: bool = False
    # reproducibility
    seed: int | None = None # Optional random seed for initialization control

    def __post_init__(self):
        # Validate and set derived parameters
        assert self.train_seq_len % 128 == 0, f"train_seq_len must be multiple of 128, got {self.train_seq_len}"
        assert self.val_seq_len % 128 == 0, f"val_seq_len must be multiple of 128, got {self.val_seq_len}"
        assert self.grad_acc_steps >= 1, f"grad_acc steps must be int >= 1"
        if self.head_dim is None:
            self.head_dim = self.model_dim // self.num_heads
        assert self.head_dim in [2 ** i for i in range(1, 10)], f"head_dim must be a power of 2, got {self.head_dim}"
        assert self.mlp_ratio > 0, f"mlp_ratio must be positive, got {self.mlp_ratio}"
        assert self.num_layers // 2 >= self.num_val_emb, \
            f"num_layers // 2 (={self.num_layers // 2}) must be greater than or equal num_val_emb (={self.num_val_emb})"
        assert self.num_layers % 2 == 0, f"Number of layers ({self.num_layers}) must be even for skip connections"

    @classmethod
    def from_args(cls):
        """Create Hyperparameters from command-line arguments."""
        parser = argparse.ArgumentParser(description="Train a GPT model with customizable hyperparameters")
        
        # Data arguments
        parser.add_argument('--train_files', type=str, help='Pattern for training data files')
        parser.add_argument('--val_files', type=str, help='Pattern for validation data files')
        parser.add_argument('--train_seq_len', type=int, help='Training sequence length')
        parser.add_argument('--val_seq_len', type=int, help='Validation sequence length')
        
        # Optimization arguments
        parser.add_argument('--val_steps', type=int, help='Number of steps to run validation for')
        parser.add_argument('--train_steps', type=int, help='Number of training iterations')
        parser.add_argument('--grad_acc_steps', type=int, help='Number of gradient accumulation steps per training iteration')
        parser.add_argument('--cooldown_frac', type=float, help='Fraction of training for learning rate cooldown')
        
        # Architecture arguments
        parser.add_argument('--tokenizer', type=str, help='Tokenizer file name in tokenizers/ directory')
        parser.add_argument('--vocab_size', type=int, help='Vocabulary size')
        parser.add_argument('--num_layers', type=int, help='Number of transformer layers')
        parser.add_argument('--num_heads', type=int, help='Number of attention heads')
        parser.add_argument('--model_dim', type=int, help='Model embedding dimension')
        parser.add_argument('--head_dim', type=int, help='Dimension per attention head')
        parser.add_argument('--mlp_ratio', type=int, help='MLP hidden dim ratio')
        parser.add_argument('--num_val_emb', type=int, help='Number of value embeddings used at initial and final layers')
        
        # Other options
        parser.add_argument('--use_fp8', type=lambda x: (str(x).lower() == 'true'), default=None, 
                            help='experimental; True on H100s (and newer?) should improve performance but seems to use more vram somehow')
        parser.add_argument('--val_loss_every', type=int, help='Evaluate validation loss every N steps')
        parser.add_argument('--save_model', type=lambda x: (str(x).lower() == 'true'), default=None, help='Save model checkpoints')
        parser.add_argument('--model_name', type=str, help='Model name for logging')
        parser.add_argument('--seed', type=int, help='Random seed for initialization control')
        
        args = parser.parse_args()
        
        # Create a base instance with defaults
        instance = cls()
        
        # Update instance with command-line arguments that were provided
        for key, value in vars(args).items():
            if value is not None:  # Only update if argument was provided
                setattr(instance, key, value)
        
        # Run post_init validations after applying CLI arguments
        instance.__post_init__()
        
        return instance, args
