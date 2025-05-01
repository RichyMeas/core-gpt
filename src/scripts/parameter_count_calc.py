import omegaconf
import hydra
from omegaconf import DictConfig, OmegaConf
from src.models.gpt_model import next_multiple_of_n

def mlp_parameter_count(cfg):
    """Calculate the number of parameters in the MLP."""
    ffn_hidden_size = cfg.model_dim * cfg.mlp_ratio
    fc1_params = cfg.model_dim * ffn_hidden_size
    fc2_params = ffn_hidden_size * cfg.model_dim
    total_params = fc1_params + fc2_params
    return total_params

def attention_parameter_count(cfg):
    """Calculate the number of parameters in the attention layer."""
    if cfg.head_dim is None:
        head_dim = cfg.model_dim // cfg.num_heads
    else:
        head_dim = cfg.head_dim
    num_heads = cfg.num_heads
    qkv_params = 3 * cfg.model_dim * (head_dim * num_heads)
    proj_params = cfg.model_dim * (head_dim * num_heads)
    lambdas = 2
    total_params = qkv_params + proj_params + lambdas
    return total_params

def word_embedding_and_output_layer_parameter_count(cfg):
    """Calculate the number of parameters in the word embedding layer."""
    vocab_size = cfg.vocab_size
    embedding_dim = cfg.model_dim
    padding = next_multiple_of_n(vocab_size, n=128)
    total_params = vocab_size * embedding_dim + (embedding_dim * padding)
    params_without_padding = 2 * vocab_size * embedding_dim
    return total_params, params_without_padding

def value_embedding_parameter_count(cfg):
    """Calculate the number of parameters in the value embedding layer."""
    vocab_size = cfg.vocab_size
    if cfg.head_dim is None:
        head_dim = cfg.model_dim // cfg.num_heads
    else:
        head_dim = cfg.head_dim
    embedding_dim = cfg.num_heads * head_dim
    num_val_emb = cfg.num_val_emb
    total_params = num_val_emb * vocab_size * embedding_dim
    return total_params

def transformer_block_parameter_count(cfg):
    attention_params = attention_parameter_count(cfg)
    mlp_params = mlp_parameter_count(cfg)
    lambdas = 2
    total_params = attention_params + mlp_params + lambdas
    return total_params

@hydra.main(version_base=None, config_path="./configs", config_name="train_gpt.yaml")
def main(cfg):
    OmegaConf.resolve(cfg)
    total_parameters = 0
    word_embed_and_output_layer_params, without_padding = word_embedding_and_output_layer_parameter_count(cfg)
    value_emb_params = value_embedding_parameter_count(cfg)
    skip_weights_params = cfg.num_layers // 2
    total_parameters += (transformer_block_parameter_count(cfg) * cfg.num_layers) + word_embed_and_output_layer_params + value_emb_params + skip_weights_params
    print(50*"=")
    print(f"Word embedding and output layer parameters: {word_embed_and_output_layer_params:,}")
    print(f"Value embedding parameters: {value_emb_params:,}")
    print(f"Per block parameters: {transformer_block_parameter_count(cfg):,}")
    print(f"MLP parameters: {mlp_parameter_count(cfg):,}")
    print(f"Attention parameters: {attention_parameter_count(cfg):,}")
    print(f"Total number of parameters with padding: {total_parameters:,}")
    print(f"Total number of parameters without padding: {(total_parameters - word_embed_and_output_layer_params) + without_padding:,}")
    print(50*"=")

if __name__ == "__main__":
    main()
    