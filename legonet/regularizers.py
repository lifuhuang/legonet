# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:41:04 2016

@author: lifu
"""

from tensorflow.contrib.layers import l1_regularizer
from tensorflow.contrib.layers import l2_regularizer

_regularizers = {'l1': l1_regularizer,
                 'l2': l2_regularizer}
                 
def get(name):
    """Return regularizer according to name.
    """
    
    return _regularizers.get(name, None)