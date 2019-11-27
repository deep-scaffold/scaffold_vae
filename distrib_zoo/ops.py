"""Operations"""
from torch import nn


__all__ = ['get_activation']


def get_activation(name, *args, **kwargs):
    """Get activation function by name"""
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(*args, **kwargs)
    if name == 'elu':
        return nn.ELU(*args, **kwargs)
    if name == 'selu':
        return nn.SELU(*args, **kwargs)
    raise ValueError('Activation not implemented')
