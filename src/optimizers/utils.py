# learning rate schedule: stable then decay
def get_lr(step: int, training_steps: int, cooldown_frac: float) -> float:
    if step < 0 or training_steps <= 0 or cooldown_frac <= 0:
        raise ValueError(
            f"Step, training steps, or cooldown frac is negative. "
            f"Received step={step}, training_steps={training_steps}, cooldown_frac={cooldown_frac}"
        )

    x = step / training_steps # progress in training
    if x > 1:
        raise ValueError (
            f"step / training_steps is greater than 1. "
            f"Received {step} / {training_steps} = {x}"
        )
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / cooldown_frac
        return w * 1.0 + (1 - w) * 0.1