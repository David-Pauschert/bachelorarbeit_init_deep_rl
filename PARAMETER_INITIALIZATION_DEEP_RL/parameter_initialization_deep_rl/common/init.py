import math
from torch import nn
from torch import Tensor
import torch as th
import numpy as np


def init_weights(self, module: nn.Module, weight_init: str = "xavier_normal", gain: float = 1.) -> None:
    """Initialize the weights of a neural network layer

    Args:
        module (nn.Module): module that represents a layer of the NN
        weight_init (str, optional): method used to initialize the weights. Defaults to "xavier_normal".
        gain (float, optional): scaling factor for the standard deviation. Defaults to 1..
    """
    if(weight_init == "constant"):
        nn.init.constant_(module.weight, gain * 1.)
    if(weight_init == "lecun_normal"):
        lecun_normal_(module.weight, gain=gain)
    if(weight_init == "lecun_uniform"):
        lecun_uniform_(module.weight, gain=gain)
    if(weight_init == "xavier_normal"):
        nn.init.xavier_normal_(module.weight, gain=gain)
    if(weight_init == "xavier_uniform"):
        nn.init.xavier_uniform_(module.weight, gain=gain)
    if(weight_init == "kaiming_normal"):
        kaiming_normal_(module.weight, gain=gain)
    if(weight_init == "kaiming_uniform"):
        kaiming_uniform_(module.weight, gain=gain)
    if(weight_init == "orthogonal"):
        nn.init.orthogonal_(module.weight, gain=gain)
    if(weight_init == "sparse"):
        nn.init.sparse_(module.weight, 0.2)
    if(weight_init == "nguyen_widrow"):
        nguyen_widrow_(module.weight, module.bias, gain=gain)
    #print(module._get_name(), module.weight.size(), module.weight)


def init_bias(self, module: nn.Module, fan_in, bias_init: str = "lecun_uniform") -> None:
    """Initialize the bias of a neural network layer

    Args:
        module (nn.Module): module that represents a layer of the NN
        fan_in (_type_): number of input neurons to the layer
        bias_init (str, optional): method used to initialize the bias. Defaults to "lecun_uniform".
    """
    if(bias_init == "random_uniform"):
        random_uniform(module.bias, fan_in=fan_in)
    if(bias_init == "zeros"):
        nn.init.zeros_(module.bias)
    if(bias_init == "0.01"):
        nn.init.constant_(module.bias, 0.01)
    if(bias_init == "nguyen_widrow"):
        pass
    #print(module.bias.size(), module.bias)


def lecun_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    """Initialize tensor with lecun heuristic sampled from a normal distribution

    Args:
        tensor (Tensor): tensor to be initialized
        gain (float, optional): gain (scaling) factor for the standard deviation. Defaults to 1..

    Returns:
        Tensor: tensor initialized with lecun heuristic
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1.0 / float(fan_in))
    return nn.init._no_grad_normal_(tensor, 0., std)


def lecun_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    """Initialize tensor with lecun heuristic sampled from a uniform distribution

    Args:
        tensor (Tensor): tensor to be initalized
        gain (float, optional): gain (scaling) factor for the standard deviation. Defaults to 1..

    Returns:
        Tensor: tensor initialized with lecun heuristic
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1.0 / float(fan_in))
    # Calculate uniform bounds from standard deviation
    a = math.sqrt(3.0) * std
    return nn.init._no_grad_uniform_(tensor, -a, a)


def random_uniform(tensor: Tensor, fan_in, gain: float = 1.):
    """Initialize tensor with heuristic sampled from random uniform description [-1/fan_in,1/fan_out]

    Args:
        tensor (Tensor): tensor to be initialized
        fan_in (_type_): number of input neurons to the layer
        gain (float, optional): gain (scaling) factor for the standard deviation. Defaults to 1.

    Returns:
        tensor initialized with random heuristic
    """
    a = math.sqrt(1 / float(fan_in))
    return nn.init._no_grad_uniform_(tensor, -a, a)


def kaiming_normal_(tensor: Tensor, mode: str = "fan_in", gain: float = 1.):
    """Initialize tensor with kaiming heuristic sampled from a normal distribution

    Args:
        tensor (Tensor): tensor to be initialized
        gain (float, optional): gain (scaling) factor for the standard deviation. Defaults to 1..

    Returns:
        Tensor: tensor initialized with kaiming heuristic
    """
    fan = nn.init._calculate_correct_fan(tensor, mode)
    std = gain * math.sqrt(2 / fan)
    with th.no_grad():
        return tensor.normal_(0, std)


def kaiming_uniform_(tensor: Tensor, mode: str = "fan_in", gain: float = 1.):
    """Initialize tensor with kaiming heuristic sampled from a uniform distribution

    Args:
        tensor (Tensor): tensor to be initialized
        gain (float, optional): gain (scaling) factor for the standard deviation. Defaults to 1..

    Returns:
        Tensor: tensor initialized with kaiming heuristic
    """
    fan = nn.init._calculate_correct_fan(tensor, mode)
    std = gain * math.sqrt(2 / fan)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    with th.no_grad():
        return tensor.uniform_(-bound, bound)
