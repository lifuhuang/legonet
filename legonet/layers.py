# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:17:39 2016

@author: lifu
"""


import tensorflow as tf

from . import initializers
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
    
    
    def __init__(self, size, name=None, activation_fn=None, 
                 weight_init='xavier', bias_init='constant', 
                 weight_reg=None, bias_reg=None, has_bias=True):
        """Initializes a new FullyConnected instance.
        """
        
        super(FullyConnected, self).__init__(size, name)
        self.has_bias = has_bias
        self.prev_layer = None
        self.W = None
        self.b = None
        self.output = None
        
        self._activation_fn = activations.get(activation_fn) or activation_fn
        self._weight_init = initializers.get(weight_init) or weight_init
        self._bias_init = initializers.get(bias_init) or weight_init
        self._weight_reg = weight_reg
        self._bias_reg = bias_reg
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        self.prev_layer = prev_layer
        with tf.variable_scope(self.name):
            with tf.variable_scope('affine_transformation'):
                self.W = tf.get_variable('W', 
                                         [self.prev_layer.size, self.size], 
                                         initializer=self._weight_init(),
                                         regularizer=self._weight_reg)
                if self.has_bias:
                    self.b = tf.get_variable('b', [self.size], 
                                             initializer=self._bias_init(),
                                             regularizer=self._bias_reg)
            
                self.output = tf.matmul(prev_layer.output, self.W)
                if self.has_bias:
                    self.output = tf.add(self.output, self.b)
            
            with tf.variable_scope('activation_function'):
                if self._activation_fn is not None:
                    self.output = self._activation_fn(self.output)
            
class Output(Layer):
    """Output layer that wraps around another layer.
    """
    
    
    def __init__(self, size, name, dtype, 
                 output_fn=None, loss_fn=None, target_size=None,
                 inner_layer=None, **kwargs):
        """Initializes a new Output instance.
        """
        
        super(Output, self).__init__(size, name)
        
        if output_fn is None:
            output_fn = 'softmax'
            
        if loss_fn is None:        
            loss_fn = 'sparse_softmax_cross_entropy'
        
        if inner_layer is None:
            self.inner_layer = FullyConnected(size, 'inner_layer', **kwargs)
        else:
            self.inner_layer = inner_layer
        
        self.size = size
        self.name = name
        self.dtype = dtype
        self.target_size = target_size or self.size
        self._output_fn = activations.get(output_fn) or output_fn
        self._loss_fn = objectives.get(loss_fn) or loss_fn
        
        self.loss = None
        self.targets = None
        self.output = None
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_scope(self.name):
            self.inner_layer.build(prev_layer)            
            with tf.variable_scope('output'):
                self.output = self._output_fn(self.inner_layer.output)
            with tf.variable_scope('loss'):      
                self.targets = tf.placeholder(
                    self.dtype, [None, self.target_size], 'targets')
                self.loss = self._loss_fn(self.inner_layer.output, 
                                          self.targets)

class Input(Layer):
    """Input layer.
    """
    
        
    def __init__(self, size, name, dtype=tf.float32):
        """Initializes a new Input instance.
        """
        
        super(Input, self).__init__(size, name)
        self.dtype=dtype
        
    def build(self, prev_layer=None):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_scope(self.name):
            self.output = tf.placeholder(self.dtype, 
                                         [None, self.size], 'input')
                                     