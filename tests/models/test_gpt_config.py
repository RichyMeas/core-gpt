import pytest 
import re
from src.models.gpt_config import Hyperparameters

def test_train_seq_len_exception():
    train_seq_len = 127
    error_msg = f"train_seq_len must be multiple of 128, got {train_seq_len}"
    
    with pytest.raises(AssertionError, match=error_msg):
        h_params = Hyperparameters(train_seq_len=train_seq_len)

def test_val_seq_len_exception():
    val_seq_len = 127
    error_msg = f"val_seq_len must be multiple of 128, got {val_seq_len}"

    with pytest.raises(AssertionError, match=error_msg):
        h_params = Hyperparameters(val_seq_len=val_seq_len)

def test_grad_acc_steps_less_than_one_exception():
    grad_acc_steps = -1
    error_msg = f"grad_acc steps must be int >= 1"

    with pytest.raises(AssertionError, match=error_msg):
        h_params = Hyperparameters(grad_acc_steps=grad_acc_steps)

def test_head_dim_is_none():
    head_dim = None
    h_params = Hyperparameters(head_dim=head_dim)

    assert h_params.head_dim == 64

def test_invalid_head_dim_powers_of_two_exception():
    head_dim = 0
    error_msg = f"head_dim must be a power of 2, got {head_dim}"

    with pytest.raises(AssertionError, match=error_msg):
        h_params = Hyperparameters(head_dim=head_dim)

def test_mlp_ratio_less_than_zero_exception():
    mlp_ratio = -1
    error_msg = f"mlp_ratio must be positive, got {mlp_ratio}"

    with pytest.raises(AssertionError, match=error_msg):
        h_params = Hyperparameters(mlp_ratio=mlp_ratio)

def test_layers_div_by_two_less_than_val_emb_exception():
    num_layers = 4
    num_val_emb = 6
    error_msg = f"num_layers // 2 (={num_layers // 2}) must be greater than or equal num_val_emb (={num_val_emb})"
    
    with pytest.raises(AssertionError, match=re.escape(error_msg)):
        h_params = Hyperparameters(num_layers=num_layers, num_val_emb=num_val_emb)

def test_layers_divided_by_two_has_remainder():
    num_layers = 41
    error_msg = f"Number of layers ({num_layers}) must be even for skip connections"
    
    with pytest.raises(AssertionError, match=re.escape(error_msg)):
        h_params = Hyperparameters(num_layers=num_layers)