# learning rate schedule: stable then decay
def get_lr(step: int, training_steps: int, cooldown_frac: float) -> float:
    x = step / training_steps # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / cooldown_frac
        return w * 1.0 + (1 - w) * 0.1