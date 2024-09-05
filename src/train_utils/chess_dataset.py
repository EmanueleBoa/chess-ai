import numpy as np
from torch.utils.data import Dataset

from src.models.utils import get_variable_from_np_array


class ChessDataset(Dataset):
    def __init__(self, move, target):
        self.move = move
        self.target = target

    def __len__(self):
        return len(self.move)

    def __getitem__(self, index):
        return self.move[index], self.target[index]

    @classmethod
    def from_numpy(cls, X: np.ndarray, y: np.ndarray):
        return cls(get_variable_from_np_array(X), get_variable_from_np_array(y))
