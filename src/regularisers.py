# src/regularisers.py
import torch
import torch.nn as nn


def l2_param_loss(model: nn.Module) -> torch.Tensor:
    reg = 0.0
    for p in model.parameters():
        if p.requires_grad:
            reg = reg + (p ** 2).sum()
    return reg


