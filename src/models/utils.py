import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset


class ChessDataSet(Dataset):
    def __init__(self, move, target):
        self.move = move
        self.target = target

    def __len__(self):
        return len(self.move)

    def __getitem__(self, index):
        return self.move[index], self.target[index]


def get_variable_from_np_array(np_array, requires_grad=False):
    return Variable(torch.from_numpy(np_array), requires_grad=requires_grad).type(torch.FloatTensor)


def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)
