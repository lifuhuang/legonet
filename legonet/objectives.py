# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:09:59 2016

@author: lifu
"""


import tensorflow as tf


def softmax_cross_entropy(logits, labels):
    """Cross-entropy error for softmax output layer.
    :param logits: Unscaled log probabilities.
    :param labels: Each row must be a valid probability distribution.
    :return: A scalar indicating the cross-entropy error.
    """
    
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))


def sparse_softmax_cross_entropy(logits, labels):
    """Cross-entropy error for softmax output layer with one-hot target.
    :param logits: Unscaled log probabilities.
    :param labels: Each element must be a class index.
    :return: A scalar indicating the cross-entropy error.
    """
    
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))


def sigmoid_cross_entropy(logits, targets):
    """Cross-entropy error for sigmoid output layer.
    :param logits: The `Tensor` passed to logistic function.
    :param targets: A `Tensor` of the same shape as `logits`.
    :return: A scalar indicating the cross-entropy error.
    """
    
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, targets))


def mean_square(output, targets):
    """Mean square error.
    :param output: The output `Tensor` of model.
    :param targets: A `Tensor` of the same size as `output`.
    :return: A scalar indicating the mean-square error.
    """
    
    return tf.reduce_mean((output - targets) ** 2)

_objectives = {'softmax_cross_entropy': softmax_cross_entropy,
               'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
               'mean_square': mean_square,
               'sigmoid_cross_entropy': sigmoid_cross_entropy}


def get(name):
    """Return loss according to name.
    :param name: The name of the loss function.
    :return: The loss function corresponding to `name`.
    """

    if name not in _objectives:
        raise ValueError("Unknown objective function: {0}".format(name))
    return _objectives[name]
