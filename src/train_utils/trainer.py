from typing import Optional

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models import ResNet
from src.train_utils import ChessDataset


class Trainer:
    def __init__(self, batch_size: Optional[int] = 1, accumulation_steps: Optional[int] = 1):
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.criterion = nn.BCELoss()

    def train_iteration(self, model: ResNet, optimizer: Adam, train_data: ChessDataset) -> float:
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
        model.train()
        running_loss = 0
        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            running_loss += loss
            loss.backward()
            if (i + 1) % self.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        running_loss /= len(train_data)
        return running_loss
