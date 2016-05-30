# -*- coding: utf-8 -*-
"""
This module contains various kinds of objective functions used as loss functions for neural networks.
"""


import tensorflow as tf


def softmax_cross_entropy(logits, targets):
    """Cross-entropy error for softmax output layer.

    Args:
        logits: Unscaled log probabilities.
        targets: Each row must be either a valid probability distribution or all zeros. If all zeros, the
        corresponding loss term will not be added to the result.

    Returns:
        A scalar indicating the cross-entropy error.

    """
    
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, targets))


def sparse_softmax_cross_entropy(logits, labels):
    """Cross-entropy error for softmax output layer with one-hot target.

    Args:
        logits: Unscaled log probabilities.
        labels: Each element must be a index within range [0, n_classes - 1] or -1. If -1, the corresponding loss term
        will not be added to the result.

    Returns:
        A scalar indicating the cross-entropy error.

    """
    
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))


def sigmoid_cross_entropy(logits, targets):
    """Cross-entropy error for sigmoid output layer.

    Args:
        logits: The `Tensor` passed to logistic function.
        targets: A `Tensor` of the same shape as `logits`. Must be either a value in the range [0, 1] or -1. If -1, the
        corresponding loss term will not be added to the result.


    Returns:
        A scalar indicating the summed cross-entropy error of all input samples with target not being -1.

    """

    logits = tf.reshape(logits, [-1])
    targets = tf.reshape(targets, [-1])
    all_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)
    # TODO: Waiting for update from TensorFlow, so that this can be handled more gracefully.
    valid_losses = tf.boolean_mask(all_losses, tf.not_equal(targets, -1))
    return tf.reduce_mean(valid_losses)


def mean_square(output, targets):
    """Mean square error.

    Args:
        output: The output `Tensor` of model.
        targets: A `Tensor` of the same size as `output`.

    Returns:
        A scalar indicating the mean-square error.

    """
    
    return tf.reduce_mean((output - targets) ** 2)

_objectives = {'softmax_cross_entropy': softmax_cross_entropy,
               'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
               'mean_square': mean_square,
               'sigmoid_cross_entropy': sigmoid_cross_entropy}


def get(name):
    """Returns the objective function according to name.

    Args:
        name: The name of the loss function.

    Returns:
        The loss function corresponding to `name`.

    """

    if name not in _objectives:
        raise ValueError("Unknown objective function: {0}".format(name))
    return _objectives[name]
