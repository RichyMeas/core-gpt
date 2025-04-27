import pytest
from src.optimizers.utils import get_lr

def test_step_count_negative_exception():
    step = -1
    training_steps = 1
    cooldown_frac = 0.02
    error_msg = f"Step, training steps, or cooldown frac is negative. Received step={step}, training_steps={training_steps}, cooldown_frac={cooldown_frac}"
    
    with pytest.raises(ValueError, match=error_msg):
        get_lr(step, training_steps, cooldown_frac) 

def test_training_steps_negative_exception():
    step = 1
    training_steps = -1
    cooldown_frac = 0.02
    error_msg = f"Step, training steps, or cooldown frac is negative. Received step={step}, training_steps={training_steps}, cooldown_frac={cooldown_frac}"
    
    with pytest.raises(ValueError, match=error_msg):
        get_lr(step, training_steps, cooldown_frac)

def test_cooldown_frac_negative_exception():
    step = 1
    training_steps = 1
    cooldown_frac = -0.02
    error_msg = f"Step, training steps, or cooldown frac is negative. Received step={step}, training_steps={training_steps}, cooldown_frac={cooldown_frac}"
    
    with pytest.raises(ValueError, match=error_msg):
        get_lr(step, training_steps, cooldown_frac)

def test_step_ratio_greater_than_one_exception():
    step = 20
    training_steps = 10
    cooldown_frac = 0.02
    x = step/training_steps
    error_msg = f"step / training_steps is greater than 1. Received {step} / {training_steps} = {x}"

    with pytest.raises(ValueError, match=error_msg):
        get_lr(step,training_steps, cooldown_frac)

def test_lr_equals_one():
    step = 20
    training_steps = 30
    cooldown_frac = 0.02
    
    lr = get_lr(step, training_steps, cooldown_frac)

    assert lr == 1.0

def test_lr_with_cooldown():
    step = 10
    training_steps = 40
    cooldown_frac = 0.9
    
    lr = get_lr(step, training_steps, cooldown_frac)

    assert lr == 0.85
