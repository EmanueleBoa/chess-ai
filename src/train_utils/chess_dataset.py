import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    @classmethod
    def from_numpy(cls, inputs: np.ndarray, targets: np.ndarray):
        return cls(
            Variable(torch.Tensor(inputs)),
            Variable(torch.Tensor(targets))
        )
