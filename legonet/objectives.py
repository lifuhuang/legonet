# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:09:59 2016

@author: lifu
"""

import tensorflow as tf

def softmax_cross_entropy(logits, labels):
    """Cross-entropy error for softmax output layer.
    """
    
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels))
     
     
def sparse_softmax_cross_entropy(logits, labels):
    """Cross-entropy error for softmax output layer with one-hot target.
    """
    
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels))

def sigmoid_cross_entropy(logits, targets):
    """Cross-entropy error for sigmoid output layer.
    """
    
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits, targets))

def mean_square(output, labels):
    """Mean square error.
    """
    
    return tf.reduce_mean((output - labels) ** 2)
    
_objectives = {'softmax_cross_entropy': softmax_cross_entropy,
               'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
               'mean_square': mean_square,
               'sigmoid_cross_entropy': sigmoid_cross_entropy}

def get(name):
    """Return objective according to name.
    """
    
    if name not in _objectives:
        raise ValueError('Unknown objective function: {0}'.format(name))
    return _objectives[name]