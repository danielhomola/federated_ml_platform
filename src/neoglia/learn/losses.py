"""
A few example, compilable loss functions for neural networks.
"""

import torch
import torch.nn.functional as F


@torch.jit.script
def cross_entropy(pred, target):
    """
    Compiled loss function for multi-class classification.
    Expects tensor of probabilities and same shaped tensor with true label = 1.
    """
    return F.nll_loss(input=pred, target=target)


@torch.jit.script
def binary_cross_entropy(pred, target):
    """
    Compiled loss function for single class classification.
    Expects tensor of probabilities and same shaped tensor with true label = 1.
    """
    return F.binary_cross_entropy(input=pred, target=target)


@torch.jit.script
def mse(pred, target):
    """
    Compiled loss function for univariate regression problems.
    Expects tensor of predicted values and same shaped tensor with true values.
    """
    return F.mse_loss(input=pred, target=target)
