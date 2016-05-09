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
    
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

def mean_square_error(output, labels):
    """Mean square error.
    """
    
    return tf.reduce_mean(tf.reduce_sum((output - labels) ** 2))
    
_objectives = {'softmax_cross_entropy': softmax_cross_entropy,
               'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
               'mean_square_error': mean_square_error}

def get(name):
    """Return objective according to name.
    """
    
    return _objectives.get(name, None)