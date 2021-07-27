import torch
from torch import nn
from config import device, default_tensor_type

torch.set_default_tensor_type(default_tensor_type)


def power_transform(x, c=0.1):

    return (torch.pow(x, c) - 1.0) / c


class PINN_Model(nn.Module):
    def __init__(self, nodes=40, layers=2, y0=0, w_scale=None, x_scale=1):
        super(PINN_Model, self).__init__()

        self.y0 = y0
        self.w_scale = w_scale
        self.x_scale = x_scale

        self.activation = nn.GELU()

        self.seq = nn.Sequential()
        self.seq.add_module('fc_1', nn.Linear(1, nodes))
        self.seq.add_module('relu_1', self.activation)
        for i in range(layers):
            self.seq.add_module('fc_' + str(i + 2), nn.Linear(nodes, nodes))
            self.seq.add_module('relu_' + str(i + 2), self.activation)
        self.seq.add_module('fc_last', nn.Linear(nodes, self.y0.shape[1]))
        # self.seq.add_module('relu_last', nn.Softplus())

    def get_slope(self):
        return 1.0

    def forward(self, x):

        # return self.seq(torch.log(x / self.x_scale)) * (power_transform(x) + 10.0) * self.w_scale + self.y0
        return self.seq(torch.log(x / self.x_scale)) * (x / self.x_scale) * self.w_scale + self.y0
