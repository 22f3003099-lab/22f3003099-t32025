from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

def get_scheduler(optimizer, scheduler_type, T_0=2, step_size=3, gamma=0.5):
    if scheduler_type == "cosine":
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1)
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError("Scheduler must be one of: cosine, step")
