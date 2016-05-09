# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:17:32 2016

@author: lifu
"""

from tensorflow.python.training.training import AdamOptimizer as Adam
from tensorflow.python.training.training import AdadeltaOptimizer as Adadelta
from tensorflow.python.training.training import GradientDescentOptimizer as SGD
from tensorflow.python.training.training import AdagradOptimizer as Adagrad
from tensorflow.python.training.training import MomentumOptimizer as Momentum
from tensorflow.python.training.training import RMSPropOptimizer as RMSProp

_objectives = {'adam': Adam,
               'sgd': SGD,
               'adadelta': Adadelta,
               'adagrad': Adagrad,
               'momentum': Momentum,
               'rmsprop': RMSProp}

def get(name):
    """Return optimizer according to name.
    """
    
    return _objectives.get(name, None)

