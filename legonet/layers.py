# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:17:39 2016

@author: lifu
"""


import tensorflow as tf

from . import initializations
from . import activations
from . import objectives

class Layer(object):
    """Base class for all kinds of layers.
    """
    
    
    def __init__(self, size, name=None):
        """Initialize a new Layer instance.
        
        Since Layer is a abstract class, this method should only be called by
        its derived classes.
        """
        
        self.size = size
        self.name = name
    
            
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        raise NotImplementedError


class FullyConnected(Layer):
    """A simple fully connected feedforward layer.
    """
    
    
    def __init__(self, size, name=None, activation='identity', 
                 W_init='xavier', b_init='constant', has_bias=True):
        """Initializes a new FullyConnected instance.
        """
        
        super(FullyConnected, self).__init__(size, name)
        self.has_bias = has_bias
        self.prev_layer = None
        self.W = None
        self.b = None
        self.output = None
        
        self._z = None
        self._activation = activations.get(activation) or activation
        self._W_init = initializations.get(W_init) or W_init
        self._b_init = initializations.get(b_init) or W_init
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        self.prev_layer = prev_layer
        with tf.variable_scope(self.name):   
            self.W = tf.get_variable('W', [self.prev_layer.size, self.size], 
                                     initializer=self._W_init())
            if self.has_bias:
                self.b = tf.get_variable('b', [self.size], 
                                         initializer=self._b_init())
            
            self._z = tf.matmul(prev_layer.output, self.W)
            if self.has_bias:
                self._z = tf.add(self._z, self.b)
            
            self.output = self._activation(self._z)    
            
class Output(FullyConnected):
    """Output layer.
    """
    
    
    def __init__(self, objective, dtype, scalar_target=False, **kwargs):
        """Initializes a new Output instance.
        """
        
        super(Output, self).__init__(**kwargs)
        self.loss = None
        self.labels = None
        self.dtype=dtype
        self.scalar_target = scalar_target
        self._objective = objectives.get(objective) or objective
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        super(Output, self).build(prev_layer)
        with tf.variable_scope(self.name):
            shape = [None]
            if not self.scalar_target:
                shape += [self.size]
            self.targets = tf.placeholder(self.dtype, shape, 'targets')
            self.loss = self._objective(self._z, self.targets)
        
class Input(Layer):
    """Input layer.
    """
    
        
    def __init__(self, size, name=None, dtype=tf.float32):
        """Initializes a new Input instance.
        """
        
        super(Input, self).__init__(size, name)
        self.dtype=dtype
        
    def build(self, prev_layer=None):
        """Construct the layer in tensorflow graph.
        """
        
        self.output = tf.placeholder(self.dtype, [None, self.size], 'input')
                                     
                                     