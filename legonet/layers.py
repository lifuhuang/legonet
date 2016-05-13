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
    
    
    def __init__(self, name, output_shape, input_shape=None):
        """Initialize a new Layer instance.
        
        Since Layer is a abstract class, this method should only be called by
        its derived classes.
        """
        
        self.name = name
        self.output_shape = tf.TensorShape(output_shape)
        self.input_shape = tf.TensorShape(input_shape)
        self.output = None
    
            
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        self.input_shape = prev_layer.output_shape


class FullyConnected(Layer):
    """A simple fully connected feedforward layer.
    """
    
    
    def __init__(self, output_dim, name=None, activation_fn=None, 
                 weight_init='xavier', bias_init='constant', 
                 weight_reg=None, bias_reg=None, has_bias=True):
        """Initializes a new FullyConnected instance.
        """
        
        super(FullyConnected, self).__init__(name, output_dim)
        
        self.has_bias = has_bias
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
        
        super(FullyConnected, self).build(prev_layer)
        
        with tf.variable_scope(self.name):
            with tf.variable_scope('affine_transformation'):
                prev_size = self.input_shape.num_elements()
                cur_size = self.output_shape.num_elements()
                self.W = tf.get_variable(
                    'W', [prev_size, cur_size], 
                    initializer=self._weight_init(),
                    regularizer=self._weight_reg)
                if self.has_bias:
                    self.b = tf.get_variable('b', [cur_size],
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
    
    
    def __init__(self, output_shape, name, dtype=tf.float32, 
                 output_fn=None, loss_fn=None, target_shape=None, 
                 inner_layer=None, **kwargs):
        """Initializes a new Output instance.
        """
        
        super(Output, self).__init__(name, output_shape)
        
        if output_fn is None:
            output_fn = 'softmax'
            
        if loss_fn is None:        
            loss_fn = 'sparse_softmax_cross_entropy'
        
        if inner_layer is None:
            self.inner_layer = FullyConnected(
                self.output_shape.num_elements(), 'inner_layer', **kwargs)
        else:
            self.inner_layer = inner_layer
        
        self.name = name
        self.dtype = dtype
        self.target_shape = tf.TensorShape(target_shape) or self.output_shape
        self._output_fn = activations.get(output_fn) or output_fn
        self._loss_fn = objectives.get(loss_fn) or loss_fn
        
        self.loss = None
        self.targets = None
        self.output = None
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        super(Output, self).build(prev_layer)
        
        with tf.variable_scope(self.name):
            self.inner_layer.build(prev_layer)            
            with tf.variable_scope('output'):
                self.output = self._output_fn(self.inner_layer.output)
            with tf.variable_scope('loss'):      
                self.targets = tf.placeholder(
                    self.dtype, 
                    [None] + self.target_shape.as_list(), 
                    'targets')
                self.loss = self._loss_fn(
                    self.inner_layer.output, self.targets)

class Input(Layer):
    """Input layer.
    """
    
        
    def __init__(self, input_shape, name, dtype=tf.float32):
        """Initializes a new Input instance.
        """
        
        super(Input, self).__init__(name, input_shape, input_shape)
        
        self.dtype=dtype
        
    def build(self, prev_layer=None):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_scope(self.name):
            self.output = tf.placeholder(
                self.dtype, [None] + self.input_shape.as_list(), 'input')
                                     