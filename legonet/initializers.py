# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:10:50 2016

@author: lifu
"""


from tensorflow.contrib.layers import xavier_initializer as xavier
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier_conv2d
    
from tensorflow import truncated_normal_initializer as truncated_normal
from tensorflow import constant_initializer as constant
from tensorflow import random_normal_initializer as normal
from tensorflow import random_uniform_initializer as uniform
from tensorflow import uniform_unit_scaling_initializer as uniform_unit_scaling


def get(name):
    """Return an array filler according to name.
    :param name: the name of initializer.
    :return: the initializer corresponding to `name`.
    """
    
    if name not in _initializers:
        raise ValueError("Unknown initializer: {0}".format(name))
    return _initializers[name]


_initializers = {'xavier': xavier(),
                 'xavier_conv2d': xavier_conv2d(),
                 'constant': constant(),
                 'truncated_normal': truncated_normal(),
                 'random_normal': normal(),
                 'random_uniform': uniform(),
                 'uniform_unit_scaling': uniform_unit_scaling()}
