########################################
#        FINAL OUTPUT EXAMPLES         #
########################################

import torch
import tiktoken
import pickle
from .log_utils import print0

def sample_from_model(model, prompt, cfg, max_new_tokens=100, temperature=0.8, top_k=200):
    """Generate text samples from the model given a prompt."""
    tokenizer_config = pickle.load(open(f"tokenizers/{cfg.tokenizer}", 'rb'))
    enc = tiktoken.Encoding(
        name=cfg.tokenizer[:-4], # :-4 to remove the .pkl
        pat_str=tokenizer_config['pat_str'],
        mergeable_ranks=tokenizer_config['mergeable_ranks'],
        special_tokens={
            "<|endoftext|>": len(tokenizer_config['mergeable_ranks']),
        }
    )
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    # Encode the prompt
    input_ids = encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.int32, device="cuda")

    # Generate
    model.eval()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode and return
    return decode(y.tolist())

# Then at the end of training:
if master_process: 
    print0("-"*10 + " EXAMPLE MODEL GENERATIONS AFTER TRAINING " + "-"*10)
    prompts = [
        "Once upon a time,",
        "The meaning of life is",
        "In the year 2026,",
        "I'm a Large Language Model (LLM), which means"
    ]
    for prompt in prompts:
        continuation = sample_from_model(model, prompt, max_new_tokens=16)
        print0(continuation, console=True)