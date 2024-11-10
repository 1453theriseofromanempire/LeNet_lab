import torch
import torch.nn as nn
import torch.optim as optim


def create_optimizer(model, lr):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return optimizer