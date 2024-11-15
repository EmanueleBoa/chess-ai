import torch
from torch import nn
from torch.autograd import Variable


def get_available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_variable_from_np_array(np_array, requires_grad=False):
    return Variable(torch.from_numpy(np_array), requires_grad=requires_grad).type(torch.FloatTensor)


def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)
