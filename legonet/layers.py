# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:17:39 2016

@author: lifu
"""


import tensorflow as tf

from . import initializers
from . import activations
from . topology import Node

class Layer(Node):
    """Base class for all kinds of layers.
    """
    
    # TODO: add operators: +, &
    
    def __init__(self, name, trainable=True):
        """Initialize a new Layer instance.
        """
        
        super(Layer, self).__init__(name)
        self.trainable = trainable
        self.params = []

class FullyConnected(Layer):
    """A simple fully connected feedforward layer.
    """
    
    
    def __init__(self, name, n_output_units, activation_fn=None, 
                 weight_init=None, bias_init=None, 
                 weight_regularizer=None, bias_regularizer=None, 
                 has_bias=True, trainable=True):
        """Initializes a new FullyConnected instance.
        """
        
        if weight_init is None:
            weight_init = initializers.xavier()
        if bias_init is None:
            bias_init = initializers.constant()
            
        super(FullyConnected, self).__init__(name, trainable)
        
        self.prev_layer = None
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
            
    def build(self, graph=None, session=None, reuse=False):
        """Construct the layer in tensorflow graph.
        """
        
        if graph is None:
            graph = tf.get_default_graph()
        if session is None:
            session = tf.get_default_session()
            
        self.input = self.pred.output
        with tf.variable_scope(self.name, reuse=reuse):
            with tf.variable_scope('affine_transformation'):
                prev_size = self.input.get_shape()[1:].num_elements()
                self.W = tf.get_variable(
                    'W', [prev_size, self.n_output_units], 
                    initializer=self._weight_init,
                    regularizer=self._weight_regularizer,
                    trainable=self.trainable)
                if not reuse:
                    self.params.append(self.W)
                flat_input = tf.reshape(self.input, (-1, prev_size))
                self.output = tf.matmul(flat_input, self.W)
                    
                if self.has_bias:
                    self.b = tf.get_variable(
                        'b', 
                        [self.n_output_units], 
                        initializer=self._bias_init, 
                        regularizer=self._bias_regularizer,
                        trainable=self.trainable)
                    if not reuse:
                        self.params.append(self.b)
                    self.output = tf.nn.bias_add(self.output, self.b)
            
            if self._activation_fn is not None:
                with tf.variable_scope('activation_fn'):
                    self.output = self._activation_fn(self.output)
            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self.output)
            tf.add_to_collection(tf.GraphKeys.BIASES, self.b)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.W)
            
            tf.initialize_variables(self.params)
                    
class Convolution2D(Layer):
    """2D Convolution layer.
    """
    
    
    def __init__(self, name, filter_shape, n_output_channels, 
                 activation_fn='relu', strides=[1, 1], padding='SAME', 
                 weight_init=None, bias_init=None, 
                 weight_regularizer=None, bias_regularizer=None, 
                 use_cudnn_on_gpu=True, has_bias=True, trainable=True):
        """Initializes a new Convolution2D instance.
        """
        
        if weight_init is None:
            weight_init = initializers.xavier_conv2d()
        if bias_init is None:
            bias_init = initializers.constant(0.1)
            
        super(Convolution2D, self).__init__(name, trainable)
        
        self.filter_shape = list(filter_shape)
        self.n_output_channels = n_output_channels
        self.strides = list(strides)
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
        
    def build(self, graph=None, session=None, reuse=False):
        """Construct the layer in tensorflow graph.
        """
        
        if graph is None:
            graph = tf.get_default_graph()
        if session is None:
            session = tf.get_default_session()
            
        self.input = self.pred.output
        with tf.variable_scope(self.name, reuse=reuse):
            full_shape = (self.filter_shape + 
                [self.input.get_shape()[-1].value, self.n_output_channels])
            self.filter = tf.get_variable(
                'filter', 
                full_shape, 
                initializer=self._weight_init,
                regularizer=self._weight_regularizer,
                trainable=self.trainable)
            if not reuse:
                self.params.append(self.filter)
            self.output = tf.nn.conv2d(
                self.input, self.filter, [1] + self.strides + [1],
                self.padding, self.use_cudnn_on_gpu)
                
            if self.has_bias:
                self.bias = tf.get_variable(
                    'bias', 
                    self.n_output_channels, 
                    initializer=self._bias_init, 
                    regularizer=self._bias_regularizer,
                    trainable=self.trainable)
                if not reuse:
                    self.params.append(self.bias)
                self.output = tf.nn.bias_add(self.output, self.bias)
                
            if self._activation_fn is not None:
                self.output = self._activation_fn(self.output)
            
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self.output)
            tf.add_to_collection(tf.GraphKeys.BIASES, self.bias)
            tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.filter)
            
            tf.initialize_variables(self.params)
            
class Pooling2D(Layer):
    """Pooling layer for 2D arrays.
    """
    
    
    def __init__(self, name, pool_shape=[2, 2], strides=[2, 2], mode='max',
                 padding='VALID'):
        """Initializes a new Convolution2D instance.
        """
        
        super(Pooling2D, self).__init__(name)
        
        self.pool_shape = list(pool_shape)
        self.strides = list(strides)
        self.padding = padding
        
        if mode == 'max':
            self._pool_fn = tf.nn.max_pool
        elif mode == 'average':
            self._pool_fn = tf.nn.avg_pool
        else:
            raise ValueError('Unrecognized pooling mode {1}'.format(mode))
        
    def build(self, graph=None, session=None, reuse=False):
        """Construct the layer in tensorflow graph.
        """
        
        if graph is None:
            graph = tf.get_default_graph()
        if session is None:
            session = tf.get_default_session()
            
        self.input = self.pred.output
        with tf.variable_scope(self.name, reuse=reuse):
            self.output = self._pool_fn(
                self.input, 
                [1] + self.pool_shape + [1], 
                [1] + self.strides + [1], 
                self.padding)
        
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self.output)
            
class Input(Layer):
    """Input layer.
    """
    
        
    def __init__(self, name, input_shape, input_dtype=tf.float32):
        """Initializes a new Input instance.
        """
        
        super(Input, self).__init__(name)
        
        self.input_shape = tf.TensorShape(input_shape)
        self.input_dtype=input_dtype
        
    def build(self, graph=None, session=None, reuse=False):
        """Construct the layer in tensorflow graph.
        """
        
        if graph is None:
            graph = tf.get_default_graph()
        if session is None:
            session = tf.get_default_session()
            
        with tf.variable_scope(self.name, reuse=reuse):
            self.input = tf.placeholder(
                self.input_dtype, [None] + self.input_shape.as_list(), 'input')
            self.output = self.input
        
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self.output)
        
class Embedding(Layer):
    """Embedding layer.
    """
    # TODO: try assign and placeholder instead of tensor.
        
    def __init__(self, name, input_shape, init_values, trainable=True):
        """Initializes a new Input instance.
        """
            
        super(Embedding, self).__init__(name, trainable)
        
        self.init_values = init_values
        self.input_shape = list(input_shape)
        
        self.table_loader = None
        self.lookup_table = None
        
    def build(self, graph=None, session=None, reuse=False):
        """Construct the layer in tensorflow graph.
        """
        
        if graph is None:
            graph = tf.get_default_graph()
        if session is None:
            session = tf.get_default_session()
            
        with tf.variable_scope(self.name, reuse=reuse):            
            self.input = tf.placeholder(
                tf.int64, name='input', shape=[None] + self.input_shape)
            self.table_loader = tf.placeholder(
                tf.float32, shape=self.init_values.shape, 'loader')
            self.lookup_table = tf.get_variable(
                'lookup_table', 
                initializer=self.table_loader,
                trainable=True)
            if not reuse:
                self.params.append(self.lookup_table)
            tf.initialize_variables(self.params)
            self.output = tf.nn.embedding_lookup(self.lookup_table, self.input)
            
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, self.output)
