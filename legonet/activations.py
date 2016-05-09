# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:55:04 2016

@author: lifu
"""

from tensorflow.python.ops.nn import relu
from tensorflow.python.ops.nn import softmax
from tensorflow import sigmoid
from tensorflow import tanh

identity = lambda x: x

_activations = {'relu': relu,
                'identity': identity,
                'sigmoid': sigmoid,
                'softmax': softmax,
                'tanh': tanh}

def get(name):
    """Return activation according to name.
    """
    
    return _activations.get(name, None)