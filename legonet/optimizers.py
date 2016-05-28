# -*- coding: utf-8 -*-
"""
This module contains various kinds of optimizers used for updating parameters in neural networks.
"""


from tensorflow.python.training.training import AdamOptimizer as Adam
from tensorflow.python.training.training import AdadeltaOptimizer as Adadelta
from tensorflow.python.training.training import GradientDescentOptimizer as GradientDescent
from tensorflow.python.training.training import AdagradOptimizer as Adagrad
from tensorflow.python.training.training import MomentumOptimizer as Momentum
from tensorflow.python.training.training import RMSPropOptimizer as RMSProp
