import torch

info_tensor = torch.empty(2, dtype=torch.int32, device='cuda:0')

def set_stage(stage: int) -> None:
    info_tensor[0] = stage
    info_tensor[1] = 0