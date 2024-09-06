from typing import Optional

import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models import ResNet
from src.train_utils import ChessDataset


class Trainer:
    def __init__(self, batch_size: Optional[int] = 1):
        self.batch_size = batch_size
        self.loss_function = nn.BCELoss()

    def train_iteration(self, model: ResNet, optimizer: Adam, train_data: ChessDataset) -> float:
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
        model.train()
        running_loss = 0
        for train_in, train_out in train_loader:
            optimizer.zero_grad()
            out = model(train_in)
            loss = self.loss_function(out, train_out)
            running_loss += loss
            loss.backward()
            optimizer.step()
        running_loss /= len(train_data)
        return running_loss
