import pytest
import torch
from src.models.embeddings.rope import Rotary

def test_rope_embedding_initalization():
    dim = 64
    max_seq_len = 512
    rope = Rotary(dim, max_seq_len)
    
    assert isinstance(rope, Rotary)
    assert hasattr(rope, 'cos') and hasattr(rope, 'sin')
    assert rope.cos.shape == (max_seq_len, dim // 2)
    assert rope.sin.shape == (max_seq_len, dim // 2)
    assert rope.cos.dtype == torch.float32
    assert rope.sin.dtype == torch.float32

def test_rotary_forward_shape_and_dtype():
    dim = 8
    max_seq_len = 10
    rotary = Rotary(dim, max_seq_len)
    x_BTHD = torch.randn(2, max_seq_len, 4, dim)  # Batch size = 2, seq_len = 10, heads = 4, dim = 8
    
    output = rotary(x_BTHD)
    
    assert output.size() == (2, max_seq_len, 4, dim)
    assert output.dtype == x_BTHD.dtype

def test_rotary_forward_shape_mismatch_exception():
    dim = 8
    max_seq_len = 2
    rotary = Rotary(dim, max_seq_len)
    x_BTHD = torch.randn(2, 200, 4, dim)  # Batch size = 2, seq_len = 200
    error_msg = f"Cosine embeddings is shorter than the sequence length. RoPE expects a sequence length of {max_seq_len}, but got {x_BTHD.size(-3)}."

    # Test that the assertion error is raised
    with pytest.raises(AssertionError, match=error_msg):
        rotary(x_BTHD)
