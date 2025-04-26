import copy
import torch
from torch import distributed as dist

def warmup_kernels(model, optimizers, args, world_size):
    warmup_steps = 10
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
    for _ in range(warmup_steps):
        loss = torch.tensor([0.], device="cuda")
        for _ in range(args.grad_acc_steps):
            inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda", dtype=torch.int64)
            #torch.compiler.cudagraph_mark_step_begin()
                # TODO why does un-commenting this^ line throw an error here in the warmup but not down in training?
            step_loss = model(inputs.to(torch.int32), targets)
            loss += step_loss / args.grad_acc_steps
        loss.backward()
        if world_size > 1:
            for param in model.parameters():
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del initial_state # TODO optionally save 