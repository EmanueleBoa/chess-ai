from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import ResNet
from training.utils import ChessDataSet


class Trainer:
    def __init__(self, learning_rate: Optional[float] = 0.001, weight_decay: Optional[float] = None,
                 batch_size: Optional[int] = 1):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.loss_function = nn.BCELoss()
        self.optimizer = None

    def init_optimizer(self, model: ResNet):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def train_iteration(self, model: ResNet, X: np.ndarray, y: np.ndarray) -> float:
        train_data = ChessDataSet.from_numpy(X, y)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
        model.train()
        if self.optimizer is None:
            self.init_optimizer(model)

        running_loss = 0
        for train_in, train_out in train_loader:
            self.optimizer.zero_grad()
            out = model(train_in)
            loss = self.loss_function(out, train_out)
            running_loss += loss
            loss.backward()
            self.optimizer.step()
        running_loss /= len(train_data)
        return running_loss
