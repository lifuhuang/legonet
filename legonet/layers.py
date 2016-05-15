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
    
    
    def __init__(self, name):
        """Initialize a new Layer instance.
        
        Since Layer is a abstract class, this method should only be called by
        its derived classes.
        """
        
        self.name = name
        self.output = None
    
            
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        raise NotImplementedError


class FullyConnected(Layer):
    """A simple fully connected feedforward layer.
    """
    
    
    def __init__(self, name, n_output_units, activation_fn=None, 
                 weight_init=None, bias_init=None, 
                 weight_regularizer=None, bias_regularizer=None, 
                 has_bias=True):
        """Initializes a new FullyConnected instance.
        """
        
        if weight_init is None:
            weight_init = initializers.xavier()
        if bias_init is None:
            bias_init = initializers.constant()
            
        super(FullyConnected, self).__init__(name)
        
        self.n_output_units = n_output_units
        self.has_bias = has_bias
        self.W = None
        self.b = None
        
        if isinstance(activation_fn, str):            
            self._activation_fn = activations.get(activation_fn)
        else:
            self._activation_fn = activation_fn
            
        if isinstance(weight_init, str):     
            self._weight_init = initializers.get(weight_init)
        else:
            self._weight_init = weight_init
        
        if isinstance(bias_init, str):
            self._bias_init = initializers.get(bias_init)
        else:
            self._bias_init = bias_init
            
        self._weight_regularizer = weight_regularizer
        self._bias_regularizer = bias_regularizer
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_scope(self.name):
            with tf.variable_scope('affine_transformation'):
                prev_size = prev_layer.output.get_shape()[1:].num_elements()
                self.W = tf.get_variable(
                    'W', [prev_size, self.n_output_units], 
                    initializer=self._weight_init,
                    regularizer=self._weight_regularizer)
                flat_input = tf.reshape(prev_layer.output, (-1, prev_size))
                self.output = tf.matmul(flat_input, self.W)
                    
                if self.has_bias:
                    self.b = tf.get_variable(
                        'b', 
                        [self.n_output_units], 
                        initializer=self._bias_init, 
                        regularizer=self._bias_regularizer)
                    self.output = tf.nn.bias_add(self.output, self.b)
            
            with tf.variable_scope('activation_function'):
                if self._activation_fn is not None:
                    self.output = self._activation_fn(self.output)
                    
class Convolution2D(Layer):
    """2D Convolution layer.
    """
    
    
    def __init__(self, name, filter_height, filter_width, n_output_channels, 
                 activation_fn='relu', strides=(1, 1), padding='SAME', 
                 weight_init=None, bias_init=None, 
                 weight_regularizer=None, bias_regularizer=None, 
                 use_cudnn_on_gpu=True, has_bias=True):
        """Initializes a new Convolution2D instance.
        """
        
        if weight_init is None:
            weight_init = initializers.xavier_conv2d()
        if bias_init is None:
            bias_init = initializers.constant(0.1)
            
        super(Convolution2D, self).__init__(name)
        
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.n_output_channels = n_output_channels
        self.strides = strides
        self.padding = padding
        self.use_cudnn_on_gpu = use_cudnn_on_gpu
        self.has_bias = has_bias
                
        self.filter = None
        self.bias = None
        
        if isinstance(activation_fn, str):            
            self._activation_fn = activations.get(activation_fn)
        else:
            self._activation_fn = activation_fn
            
        if isinstance(weight_init, str):     
            self._weight_init = initializers.get(weight_init)
        else:
            self._weight_init = weight_init
        
        if isinstance(bias_init, str):
            self._bias_init = initializers.get(bias_init)
        else:
            self._bias_init = bias_init
            
        self._weight_regularizer = weight_regularizer
        self._bias_regularizer = bias_regularizer
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_scope(self.name):
            filter_shape = [self.filter_height,
                            self.filter_width,
                            prev_layer.output.get_shape()[-1].value,
                            self.n_output_channels]
            self.filter = tf.get_variable(
                'filter', filter_shape, initializer=self._weight_init,
                regularizer=self._weight_regularizer)
            self.output = tf.nn.conv2d(
                prev_layer.output, self.filter, [1] + list(self.strides) + [1],
                self.padding, self.use_cudnn_on_gpu)
                
            if self.has_bias:
                self.bias = tf.get_variable(
                    'bias', 
                    self.n_output_channels, 
                    initializer=self._bias_init, 
                    regularizer=self._bias_regularizer)
                self.output = tf.nn.bias_add(self.output, self.bias)
            
            if self._activation_fn is not None:
                self.output = self._activation_fn(self.output)

class Pooling2D(Layer):
    """Pooling layer for 2D arrays.
    """
    
    
    def __init__(self, name, pool_shape=(2, 2), strides=(2, 2), mode='max',
                 padding='VALID'):
        """Initializes a new Convolution2D instance.
        """
        
        super(Pooling2D, self).__init__(name)
        
        self.pool_shape = pool_shape
        self.strides = strides
        self.padding = padding
        
        if mode == 'max':
            self._pool_fn = tf.nn.max_pool
        elif mode == 'average':
            self._pool_fn = tf.nn.avg_pool
        else:
            raise ValueError('Unrecognized pooling mode {1}'.format(mode))
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_scope(self.name):
            self.output = self._pool_fn(
                prev_layer.output, 
                [1] + list(self.pool_shape) + [1], 
                [1] + list(self.strides) + [1], 
                self.padding)

class Output(Layer):
    """Output layer that wraps around another layer.
    """
    
    
    def __init__(self, name, output_shape, output_fn=None, loss_fn=None,
                 target_dtype=tf.float32, target_shape=None, 
                 inner_layer=None, **kwargs):
        """Initializes a new Output instance.
        """
        
        super(Output, self).__init__(name)
        
        self.output_shape = tf.TensorShape(output_shape)
        if output_fn is None:
            output_fn = 'softmax'
            
        if loss_fn is None:        
            loss_fn = 'sparse_softmax_cross_entropy'
        
        if inner_layer is None:
            self.inner_layer = FullyConnected(
                'inner_layer', self.output_shape.num_elements(), **kwargs)
        else:
            self.inner_layer = inner_layer
        
        self.name = name
        self.target_dtype = target_dtype
        self.target_shape = tf.TensorShape(target_shape) or self.output_shape
        self._output_fn = activations.get(output_fn) or output_fn
        self._loss_fn = objectives.get(loss_fn) or loss_fn
        
        self.loss = None
        self.targets = None
        
    def build(self, prev_layer):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_scope(self.name):
            self.inner_layer.build(prev_layer)            
            with tf.variable_scope('output'):
                self.output = self._output_fn(self.inner_layer.output)
            with tf.variable_scope('loss'):      
                self.targets = tf.placeholder(
                    self.target_dtype, 
                    [None] + self.target_shape.as_list(), 
                    'targets')
                self.loss = self._loss_fn(
                    self.inner_layer.output, self.targets)

class Input(Layer):
    """Input layer.
    """
    
        
    def __init__(self, name, input_shape, input_dtype=tf.float32):
        """Initializes a new Input instance.
        """
        
        super(Input, self).__init__(name)
        
        self.input_shape = tf.TensorShape(input_shape)
        self.input_dtype=input_dtype
        
    def build(self, prev_layer=None):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_scope(self.name):
            self.output = tf.placeholder(
                self.input_dtype, [None] + self.input_shape.as_list(), 'input')
                                     