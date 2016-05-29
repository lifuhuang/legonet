# -*- coding: utf-8 -*-
"""
This module contains activation functions used for layers in neural network.
"""


import tensorflow as tf


def identity(x):
    """Returns the input tensor without any change.

    Args:
        x: Input tensor.

    Returns:
        The input tensor.

    """

    return x


def relu(x):
    """Computes the rectified linear of x.

    Args:
        x: Input tensor.

    Returns:
        A `Tensor` of same shape as `x`.

    """

    return tf.nn.relu(x)


def tanh(x):
    """Computes the hyperbolic tangent of x.

    Args:
        x: Input tensor.

    Returns:
        A `Tensor` of same shape as `x`.

    """

    return tf.tanh(x)


def sigmoid(x):
    """Computes the sigmoid of x.

    Args:
        x: Input tensor.

    Returns:
        A `Tensor` of same shape as `x`.

    """

    return tf.sigmoid(x)


def softmax(x):
    """Computes the softmax of x.

    Args:
        x: Input tensor.

    Returns:
        A `Tensor` of same shape as `x`.

    """

    return tf.nn.softmax(x)


_activations = {'relu': relu,
                'identity': identity,
                'sigmoid': sigmoid,
                'softmax': softmax,
                'tanh': tanh}


def get(name):
    """Return activation according to name.

    Args:
        name: name of activation function.

    Returns:
        the activation according to `name`.

    """
    
    if name not in _activations:
        raise ValueError('Unknown activation function: {0}'.format(name))
    return _activations[name]
