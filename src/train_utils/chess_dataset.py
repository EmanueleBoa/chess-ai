import numpy as np
from torch.utils.data import Dataset

from src.models.utils import get_variable_from_np_array


class ChessDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    @classmethod
    def from_numpy(cls, X: np.ndarray, y: np.ndarray):
        return cls(get_variable_from_np_array(X), get_variable_from_np_array(y))
