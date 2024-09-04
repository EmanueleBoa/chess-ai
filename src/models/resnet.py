import numpy as np
import torch
from torch import nn
from torch.nn import functional

from src.models.utils import get_variable_from_np_array

MOVE_PLANES = 35
HISTORICAL_BOARD_PLANES = 12
RESIDUAL_BLOCKS = 6
CHANNELS = 64


class ResNet(nn.Module):
    def __init__(self, input_planes: int, residual_blocks: int, channels: int):
        super(ResNet, self).__init__()
        self.n_res = residual_blocks
        self.conv = ConvolutionalBlock(input_channels=input_planes, output_channels=channels)
        for block in range(residual_blocks):
            setattr(self, "residual_block_%i" % block, ResidualBlock(input_channels=channels, output_channels=channels))
        self.out_block = OutputBlock(input_channels=channels, output_channels=int(channels / 2))

    @classmethod
    def init_standard(cls, history_size: int = 0):
        input_planes = MOVE_PLANES + history_size * HISTORICAL_BOARD_PLANES
        return cls(input_planes, RESIDUAL_BLOCKS, CHANNELS)

    @classmethod
    def init_mini(cls, history_size: int = 0):
        input_planes = MOVE_PLANES + history_size * HISTORICAL_BOARD_PLANES
        return cls(input_planes, int(RESIDUAL_BLOCKS / 2), int(CHANNELS / 2))

    def forward(self, x):
        out = self.conv(x)
        for block in range(self.n_res):
            out = getattr(self, "residual_block_%i" % block)(out)
        out = self.out_block(out)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = self.forward(get_variable_from_np_array(x))
        out = out.cpu().detach().numpy().squeeze()
        return out

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = functional.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = functional.relu(out)
        return out


class ConvolutionalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(ConvolutionalBlock, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        out = x.view(-1, self.input_channels, 8, 8)
        out = functional.relu(self.bn1(self.conv1(out)))
        return out


class OutputBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(OutputBlock, self).__init__()
        self.out_channels = output_channels
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.fc1 = nn.Linear(output_channels * 8 * 8, output_channels * 4)
        self.fc2 = nn.Linear(output_channels * 4, 1)

    def forward(self, x):
        out = functional.relu(self.bn(self.conv(x)))
        out = out.view(-1, self.out_channels * 8 * 8)
        out = functional.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out
