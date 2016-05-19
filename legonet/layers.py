# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 23:17:39 2016

@author: lifu
"""


import tensorflow as tf

from . import initializers
from . import activations
from . topology import Node
from . import GraphKeys

class Layer(Node):
    """Abstract base class for all kinds of layers.
    
    This class is an abstract base class and is intended to be used only as an 
    interface for its derived classes. So, do not directly use it in neural
    network.
    """
    
    # TODO: add operators: +, &
    
    def __init__(self, name=None, trainable=True):
        """Initialize attributes in Layer.
        
        This method should be called in the constructor of derived classes.
        """
        
        super(Layer, self).__init__(name)
        self.trainable = trainable
        self.params = []
        
    def call(self, flow):
        """Construct the Layer in tensorflow graph.
        
        This method is intended to be implemented in derived classes.
        """
        
        raise NotImplementedError

class FullyConnected(Layer):
    """A simple fully connected feedforward layer.
    """
    
    
    def __init__(self, n_output_units, activation_fn=None, 
                 weight_init=None, bias_init=None, 
                 weight_regularizer=None, bias_regularizer=None, 
                 has_bias=True,  name=None, trainable=True):
        """Initializes a new FullyConnected instance.
        """
        
        if weight_init is None:
            weight_init = initializers.xavier()
        if bias_init is None:
            bias_init = initializers.constant()
            
        super(FullyConnected, self).__init__(name, trainable)
        
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
            
    def call(self, flow):
        """Construct the layer in tensorflow graph.
        """
            
        with tf.variable_op_scope(
            [flow], self.name, 'FC', reuse=self.reuse):
            if not self.reuse:
                prev_size = flow.get_shape()[1:].num_elements()
                self.W = tf.get_variable(
                    'W', 
                    [prev_size, self.n_output_units], 
                    initializer=self._weight_init,
                    regularizer=self._weight_regularizer,
                    trainable=self.trainable)
                self.params.append(self.W)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.W)
                    
                if self.has_bias:
                    self.b = tf.get_variable(
                        'b', 
                        [self.n_output_units], 
                        initializer=self._bias_init, 
                        regularizer=self._bias_regularizer,
                        trainable=self.trainable)
                    self.params.append(self.b)
                    tf.add_to_collection(tf.GraphKeys.BIASES, self.b)
                
                self.reuse = True
                    
            with tf.name_scope('affine'):
                flat_input = tf.reshape(flow, (-1, prev_size))
                flow = tf.matmul(flat_input, self.W)                    
                flow = tf.nn.bias_add(flow, self.b)
                tf.initialize_variables(self.params).run()
        
            if self._activation_fn is not None:
                with tf.name_scope('activation_fn'):
                    flow = self._activation_fn(flow)
                    
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow
                    
class Convolution(Layer):
    """Convolution layer for 2D arrays.
    """
    
    
    def __init__(self, filter_shape, n_output_channels, 
                 activation_fn='relu', strides=[1, 1], padding='SAME', 
                 weight_init=None, bias_init=None, 
                 weight_regularizer=None, bias_regularizer=None, 
                 use_cudnn_on_gpu=True, has_bias=True, 
                 name=None, trainable=True):
        """Initializes a new Convolution instance.
        """
        
        if weight_init is None:
            weight_init = initializers.xavier_conv2d()
        if bias_init is None:
            bias_init = initializers.constant(0.1)
            
        super(Convolution, self).__init__(name, trainable)
        
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
        
    def call(self, flow):
        """Construct the layer in tensorflow graph.
        """
            
        with tf.variable_op_scope(
            [flow], self.name, 'Conv', reuse=self.reuse):
            if not self.reuse:
                full_shape = (self.filter_shape + [flow.get_shape()[-1].value, 
                     self.n_output_channels])
                self.filter = tf.get_variable(
                    'filter', 
                    full_shape, 
                    initializer=self._weight_init,
                    regularizer=self._weight_regularizer,
                    trainable=self.trainable)
                self.params.append(self.filter)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.filter)
                
                if self.has_bias:
                    self.bias = tf.get_variable(
                        'bias', 
                        self.n_output_channels, 
                        initializer=self._bias_init, 
                        regularizer=self._bias_regularizer,
                        trainable=self.trainable)
                    self.params.append(self.bias)
                    tf.add_to_collection(tf.GraphKeys.BIASES, self.bias)
                    
                tf.initialize_variables(self.params).run()
                self.reuse = True
                    
            flow = tf.nn.conv2d(
                flow,
                self.filter, 
                [1] + self.strides + [1],
                self.padding, 
                self.use_cudnn_on_gpu)
                
            flow = tf.nn.bias_add(flow, self.bias)
                
            if self._activation_fn is not None:
                flow = self._activation_fn(flow)
            
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow
            
class Pooling(Layer):
    """Pooling layer for 2D arrays.
    """
    
    
    def __init__(self, pool_shape=[2, 2], strides=[2, 2], mode='max',
                 padding='VALID', name=None):
        """Initializes a new Pooling instance.
        """
        
        super(Pooling, self).__init__(name)
        
        self.pool_shape = list(pool_shape)
        self.strides = list(strides)
        self.padding = padding
        
        if mode == 'max':
            self._pool_fn = tf.nn.max_pool
        elif mode == 'average':
            self._pool_fn = tf.nn.avg_pool
        else:
            raise ValueError('Unrecognized pooling mode {1}'.format(mode))
        
    def call(self, flow):
        """Construct the layer in tensorflow graph.
        """
            
        with tf.variable_op_scope([flow], self.name, 'Pool', reuse=self.reuse):
            if not self.reuse:
                self.reuse = True
                
            flow = self._pool_fn(
                flow,
                [1] + self.pool_shape + [1],
                [1] + self.strides + [1],
                self.padding)
        
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow
            
class Input(Layer):
    """Input layer.
    """
    
        
    def __init__(self, input_shape, input_dtype=tf.float32, name=None):
        """Initializes a new Input instance.
        """
        
        super(Input, self).__init__(name)
        
        self.input_shape = list(input_shape)
        self.input_dtype=input_dtype
        
    def call(self, flow=None):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_op_scope(
            [flow], self.name, 'Input', reuse=self.reuse):
            if not self.reuse:
                self.reuse = True
            flow = tf.placeholder(
                self.input_dtype, [None] + self.input_shape, 'input')
            tf.add_to_collection(GraphKeys.MODEL_INPUTS, flow)
        
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow
        
class Embedding(Layer):
    """Embedding layer.
    """
        
    def __init__(self, input_shape, init_values, name=None, trainable=True):
        """Initializes a new Input instance.
        """
            
        super(Embedding, self).__init__(name, trainable)
        
        self.init_values = init_values
        self.input_shape = list(input_shape)
        
        self.table_loader = None
        self.lookup_table = None
        
    def call(self, flow=None):
        """Construct the layer in tensorflow graph.
        """
        
        with tf.variable_op_scope(
            [flow], self.name, 'Embedding', reuse=self.reuse):
            if not self.reuse:
                self.table_loader = tf.placeholder(
                    tf.float32, shape=self.init_values.shape, name='loader')
                self.lookup_table = tf.get_variable(
                    'lookup_table', 
                    initializer=self.table_loader,
                    trainable=self.trainable)
                self.params.append(self.lookup_table)
                tf.initialize_variables(self.params).run(
                    feed_dict={self.table_loader: self.init_values})
                self.reuse = True
                
            flow = tf.placeholder(
                tf.int64, [None] + self.input_shape, 'input')                           
            tf.add_to_collection(GraphKeys.MODEL_INPUTS, flow)
            flow = tf.nn.embedding_lookup(self.lookup_table, flow)
            
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow