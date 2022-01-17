import torch
from src.losses.edge_loss import EdgeL1Loss


def get_loss(loss_name: str = 'L1'):
    if loss_name == 'L1':
        print(f"got L2 loss")
        return torch.nn.MSELoss()
    elif loss_name == 'EdgeL1':
        return EdgeL1Loss()
