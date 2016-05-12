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
        logits, tf.squeeze(labels)))

def sigmoid_cross_entropy(logits, targets):
    """Cross-entropy error for sigmoid output layer.
    """
    
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        tf.squeeze(logits), tf.squeeze(targets)))

def mean_square(output, labels):
    """Mean square error.
    """
    
    print 'output', output
    print 'labels', labels
    print 'loss', tf.reduce_mean((output - labels) ** 2)
    return tf.reduce_mean((output - labels) ** 2)
    
_objectives = {'softmax_cross_entropy': softmax_cross_entropy,
               'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
               'mean_square': mean_square,
               'sigmoid_cross_entropy': sigmoid_cross_entropy}

def get(name):
    """Return objective according to name.
    """
    
    return _objectives.get(name, None)