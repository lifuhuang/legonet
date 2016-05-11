# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:41:04 2016

@author: lifu
"""

from tensorflow.contrib.layers import l1_regularizer as l1
from tensorflow.contrib.layers import l2_regularizer as l2

_regularizers = {'l1': l1,
                 'l2': l2}
                 
def get(name):
    """Return regularizer according to name.
    """
    
    return _regularizers.get(name, None)