# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:55:04 2016

@author: lifu
"""

from tensorflow.python.ops.nn import relu
from tensorflow.python.ops.nn import softmax
from tensorflow import sigmoid
from tensorflow import tanh


def identity(x):
    """Return the input tensor without any change.
    :param x: Input tensor.
    :return: The input tensor.
    """

    return x

_activations = {'relu': relu,
                'identity': identity,
                'sigmoid': sigmoid,
                'softmax': softmax,
                'tanh': tanh}


def get(name):
    """Return activation according to name.
    :param name: name of activation function.
    :return: the activation according to `name`.
    """
    
    if name not in _activations:
        raise ValueError('Unknown activation function: {0}'.format(name))
    return _activations[name]
