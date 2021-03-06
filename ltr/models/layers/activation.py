import torch
import torch.nn as nn
import torch.nn.functional as F


class MLU(nn.Module):
    r"""MLU activation
    """
    def __init__(self, min_val, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.inplace = inplace

    def forward(self, input):
        return F.elu(F.leaky_relu(input, 1/self.min_val, inplace=self.inplace), self.min_val, inplace=self.inplace)


class LeakyReluPar(nn.Module):
    r"""LeakyRelu parametric activation
    """

    def forward(self, x, a):
        return (1.0 - a)/2.0 * torch.abs(x) + (1.0 + a)/2.0 * x


class LeakyReluParDeriv(nn.Module):
    r"""Derivative of the LeakyRelu parametric activation, wrt x.
    """

    def forward(self, x, a):
        return (1.0 - a)/2.0 * torch.sign(x.detach()) + (1.0 + a)/2.0

class MyRelu(nn.Module):
    r"""Relu activation
    """

    def forward(self, x):
        return torch.abs(x)

class MyReluDeriv(nn.Module):
    r"""Relu activation
    """

    def forward(self, x):
        x_d = torch.ones_like(x)
        x_d[x < 0] = 0
        return x_d

class MySigmod(nn.Module):

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

class MySigmoidDeriv(nn.Module):

    def forward(self, x):
        return (1 / (1 + torch.exp(-x))) * (1 - 1 / (1 + torch.exp(-x)))

class BentIdentPar(nn.Module):
    r"""BentIdent parametric activation
    """
    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, a):
        return (1.0 - a)/2.0 * (torch.sqrt(x*x + 4.0*self.b*self.b) - 2.0*self.b) + (1.0 + a)/2.0 * x


class BentIdentParDeriv(nn.Module):
    r"""BentIdent parametric activation deriv
    """
    def __init__(self, b=1.0):
        super().__init__()
        self.b = b

    def forward(self, x, a):
        return (1.0 - a)/2.0 * (x / torch.sqrt(x*x + 4.0*self.b*self.b)) + (1.0 + a)/2.0

