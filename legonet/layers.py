# -*- coding: utf-8 -*-
"""
This module contains all kinds of layers which are used in a neural network model. These layers are all derived from an
abstract base class `Layer`, which itself is derived from the abstract class `Node`. These layers can either be added to
container `Node`(including `NeuralNetwork`) or be called directly in a functional style.
"""


import abc

import tensorflow as tf

from . import initializers
from . import activations
from .topology import Node
from . import GraphKeys


class Layer(Node):
    """Abstract base class for all kinds of layers.
    
    This class is an abstract base class and is intended to be used only as an
    interface for its derived classes. So, do not directly use it in neural
    network.

    Attributes:
        trainable: Indicates whether the parameters of this layer will be updated during training.
        params: List of all parameters in this `Layer`. The list will be empty before the `Layer` being called for the
        first time.
    """

    # TODO: add operators: +, &

    def __init__(self, name=None, trainable=True):
        """Initializes attributes in `Layer`.

        This method should be called in the constructor of derived classes.

        Args:
            name: name of this `Layer`. Uses default name if `None` is given.
            trainable: Indicates whether the parameters of this layer will be updated during training.

        """

        super(Layer, self).__init__(name)
        self.trainable = trainable
        self.params = []

    @abc.abstractmethod
    def __call__(self, flow):
        """Constructs the Layer in `Tensorflow` graph.
        
        This method is intended to be implemented in derived classes.

        Args:
            flow: The input `Tensor`.

        Returns:
            Output of this layer.

        """

        pass


class FullyConnected(Layer):
    """A simple fully connected layer.

    Fully connected layer (a.k.a. dense layer) performs an affine transformation to the input `Tensor` using a weight
    matrix (usually denoted as `W`) and a bias vector (usually denoted as `b`), possibly followed by an activation
    function, which is necessary except that `FullyConnected` is used as the output layer (i.e. the last layer in the
    model).
    """

    def __init__(self, n_output_units, activation_fn=None, weight_init=None, bias_init=None, weight_regularizer=None,
                 bias_regularizer=None, has_bias=True, name=None, trainable=True):
        """Initializes a new FullyConnected instance.

        Args:
            n_output_units: Number of output units.
            activation_fn: A `str`, `callable`, or `None`.
            weight_init: A `str`, `callable`, or `None`. Uses `xavier` as default if `None` is passed.
            bias_init: A `str`, `callable`, or `None`. Uses `constant` as default if `None` is passed.
            weight_regularizer: A `callable` or `None`.
            bias_regularizer: A `callable` or `None`.
            has_bias: Indicates whether there are bias units in this layer.
            name: Name of this Layer. Uses default if `None` is passed.
            trainable: Indicates whether the parameters of this layer will be updated during training.

        """

        if activation_fn is None:
            activation_fn = activations.identity
        if weight_init is None:
            weight_init = initializers.xavier()
        if bias_init is None:
            bias_init = initializers.constant()

        super(FullyConnected, self).__init__(name, trainable)

        self._n_output_units = n_output_units
        self._has_bias = has_bias
        self._weight = None
        self._bias = None

        if isinstance(activation_fn, str):
            self._activation_fn = activations.get(activation_fn)
        elif callable(activation_fn):
            self._activation_fn = activation_fn
        else:
            raise ValueError("activation_fn can only be a str, callable, or None.")

        if isinstance(weight_init, str):
            self._weight_init = initializers.get(weight_init)
        elif callable(weight_init):
            self._weight_init = weight_init
        else:
            raise ValueError("weight_init can only be a str, callable, or None.")

        if isinstance(bias_init, str):
            self._bias_init = initializers.get(bias_init)
        elif callable(bias_init):
            self._bias_init = bias_init
        else:
            raise ValueError("bias_init can only be a str, a callable, or None.")

        self._weight_regularizer = weight_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, flow):
        """Applies this `FullyConnected` to the input `Tensor` and returns the output `Tensor`.

        Args:
            flow: The input `Tensor`.

        Returns:
            Output of this layer.

        """

        with tf.variable_op_scope([flow], self.name, 'FC', reuse=self.reuse):
            prev_size = flow.get_shape()[1:].num_elements()
            if not self.reuse:
                self._weight = tf.get_variable(
                    'W',
                    [prev_size, self._n_output_units],
                    initializer=self._weight_init,
                    regularizer=self._weight_regularizer,
                    trainable=self.trainable)
                self.params.append(self._weight)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, self._weight)

                if self._has_bias:
                    self._bias = tf.get_variable(
                        'b',
                        [self._n_output_units],
                        initializer=self._bias_init,
                        regularizer=self._bias_regularizer,
                        trainable=self.trainable)
                    self.params.append(self._bias)
                    tf.add_to_collection(tf.GraphKeys.BIASES, self._bias)

                self.reuse = True

            with tf.name_scope('affine'):
                flat_input = tf.reshape(flow, (-1, prev_size))
                flow = tf.matmul(flat_input, self._weight)
                flow = tf.nn.bias_add(flow, self._bias)
                tf.initialize_variables(self.params).run()

            if self._activation_fn is not None:
                with tf.name_scope('activation_fn'):
                    flow = self._activation_fn(flow)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow


class Convolution(Layer):
    """Convolution layer for 2D arrays.

    Convolution layer is the core layer in Convolutional Neural Network. It performs a 2D convolution operation using a
    2D filter (a.k.a convolution kernel) and adds a bias array (usually denoted as `b`) to the input `Tensor`, possibly
    followed by an activation function (often ReLU in latest research).
    """

    def __init__(self, filter_shape, n_output_channels, activation_fn='relu', strides=None, padding='SAME',
                 weight_init=None, bias_init=None, weight_regularizer=None, bias_regularizer=None,
                 use_cudnn_on_gpu=True, has_bias=True, name=None, trainable=True):
        """Initializes a new Convolution instance.

        Args:
            filter_shape: A sequence with two elements.
            n_output_channels: Number of output channels.
            activation_fn: A `str`, `callable`, or `None`.
            strides: A sequence with two elements. Defaults to `[1, 1]`.
            padding: Either `SAME` or `VALID`.
            weight_init: A `str`, `callable`, or `None`. Uses `xavier_conv2d` as default if `None` is passed.
            bias_init: A `str`, `callable`, or `None`. Uses `constant` as default if `None` is passed.
            weight_regularizer: A `callable` or `None`.
            bias_regularizer: A `callable` or `None`.
            use_cudnn_on_gpu: Indicates whether the convolution operation uses cudnn on GPU.
            has_bias: Indicates whether there are bias units in this layer.
            name: Name of this Layer. Uses default name if `None` is passed.
            trainable: Indicates whether the parameters of this layer will be updated during training.

        """

        if strides is None:
            strides = [1, 1]
        if weight_init is None:
            weight_init = initializers.xavier_conv2d()
        if bias_init is None:
            bias_init = initializers.constant(0.1)

        super(Convolution, self).__init__(name, trainable)

        self._filter_shape = list(filter_shape)
        self._n_output_channels = n_output_channels
        self._strides = list(strides)
        self._padding = padding
        self._use_cudnn_on_gpu = use_cudnn_on_gpu
        self._has_bias = has_bias

        self.filter = None
        self.bias = None

        if isinstance(activation_fn, str):
            self._activation_fn = activations.get(activation_fn)
        elif callable(activation_fn):
            self._activation_fn = activation_fn
        else:
            raise ValueError("activation_fn should be either a str or callable.")

        if isinstance(weight_init, str):
            self._weight_init = initializers.get(weight_init)
        elif callable(weight_init):
            self._weight_init = weight_init
        else:
            raise ValueError("weight_init should be either a str or callable.")

        if isinstance(bias_init, str):
            self._bias_init = initializers.get(bias_init)
        elif callable(bias_init):
            self._bias_init = bias_init
        else:
            raise ValueError("bias_init should be either a str or callable")

        self._weight_regularizer = weight_regularizer
        self._bias_regularizer = bias_regularizer

    def __call__(self, flow):
        """Applies this layer to the input `Tensor` and returns the output `Tensor`.

        Args:
            flow: The input `Tensor`.

        Returns:
            Output of this layer.

        """

        with tf.variable_op_scope([flow], self.name, 'Conv', reuse=self.reuse):
            if not self.reuse:
                full_shape = self._filter_shape + [flow.get_shape()[-1].value, self._n_output_channels]
                self.filter = tf.get_variable(
                    'filter',
                    full_shape,
                    initializer=self._weight_init,
                    regularizer=self._weight_regularizer,
                    trainable=self.trainable)
                self.params.append(self.filter)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, self.filter)

                if self._has_bias:
                    self.bias = tf.get_variable(
                        'bias',
                        self._n_output_channels,
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
                [1] + self._strides + [1],
                self._padding,
                self._use_cudnn_on_gpu)

            flow = tf.nn.bias_add(flow, self.bias)

            if self._activation_fn is not None:
                flow = self._activation_fn(flow)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow


class Pooling(Layer):
    """Pooling layer for 2D arrays.

    Pooling layer is one of the most important components in a convolutional neural network. It is used for reducing the
    size of the input tensor, so as to lower the argument number and to reduce redundant information. The process is to
    generate one scalar from every fix-sized area (called `pool`) in the input `Tensor` and combine all scalars to a new
    `Tensor` as the output of pooling layer. There are two choices for the pooling strategy: `max pooling` and `average
    pooling`.
    """

    def __init__(self, pool_shape=None, strides=None, mode='max', padding='VALID', name=None):
        """Initializes a new Pooling instance.

        Args:
            pool_shape: A sequence with two elements. Defaults to [2, 2].
            strides: A sequence with two elements. Defaults to [2, 2].
            mode: Either `max` or `average`.
            padding: Either `SAME` or `VALID`.
            name: The name of this layer. Uses default name if `None` is given.

        """

        super(Pooling, self).__init__(name)

        if pool_shape is None:
            pool_shape = [2, 2]
        if strides is None:
            strides = [2, 2]
        self._pool_shape = list(pool_shape)
        self._strides = list(strides)
        self._padding = padding

        if mode == 'max':
            self._pool_fn = tf.nn.max_pool
        elif mode == 'average':
            self._pool_fn = tf.nn.avg_pool
        else:
            raise ValueError('Unrecognized pooling mode {0}'.format(mode))

    def __call__(self, flow):
        """Applies pooling layer to the input `Tensor` and returns the output `Tensor`.

        Args:
            flow: The input `Tensor`.

        Returns:
            Output of this layer.

        """

        with tf.variable_op_scope([flow], self.name, 'Pool', reuse=self.reuse):
            if not self.reuse:
                self.reuse = True

            flow = self._pool_fn(
                flow,
                [1] + self._pool_shape + [1],
                [1] + self._strides + [1],
                self._padding)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow


class Input(Layer):
    """Input layer takes the input `Tensor` and pass it to the subsequent layers.

    This layer performs no transformation to the input and is used only for the purpose of placeholder.
    """

    def __init__(self, input_shape, input_dtype=tf.float32, name=None):
        """Initializes a new Input instance.

        Args:
            input_shape: A sequence or an `int`.
            input_dtype: The data type of input. Defaults to `float32`.
            name: The name of this layer. Uses default name if `None` is passed.

        """

        if isinstance(input_shape, int):
            input_shape = [input_shape]

        super(Input, self).__init__(name)

        self._input_shape = list(input_shape)
        self._input_dtype = input_dtype

    def __call__(self, flow=None):
        """Constructs the layer in `Tensorflow` graph.

        Args:
            flow: This argument is ignored. (Default value = None)

        Returns:
            Output of this layer.

        """

        with tf.variable_op_scope([], self.name, 'Input', reuse=self.reuse):
            if not self.reuse:
                self.reuse = True
            flow = tf.placeholder(self._input_dtype, [None] + self._input_shape, 'input')
            tf.add_to_collection(GraphKeys.MODEL_INPUTS, flow)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow


class Embedding(Layer):
    """Embedding layer uses the underlying mechanism provided by `TensorFlow` to parallel lookups on a list of tensors.

    It is very useful for tasks where the input of (subsequent) model is just a list of tensors chosen from a predefined
    table (e.g. word vectors in NLP tasks), and is especially useful when you want to fine-tune this table during the
    training process by enabling error message being propagated to tensors in this table.
    """

    def __init__(self, input_shape, init_values, name=None, trainable=True):
        """Initializes a new Input instance.

        Args:
            input_shape: A sequence with two elements.
            init_values: A array used to initialize lookup table.
            name: The name of this layer. Uses default name if `None` is passed.
            trainable: Indicates whether the parameters of this layer will be updated during training.

        """

        super(Embedding, self).__init__(name, trainable)

        self._init_values = init_values
        self._input_shape = list(input_shape)

        self._table_loader = None
        self._lookup_table = None

    def __call__(self, flow=None):
        """Constructs the layer in `Tensorflow` graph.

        Args:
            flow: This argument is ignored. (Default value = None)

        Returns:
            Output of this layer.

        """

        with tf.variable_op_scope([flow], self.name, 'Embedding', reuse=self.reuse):
            if not self.reuse:
                self._table_loader = tf.placeholder(tf.float32, shape=self._init_values.shape, name='loader')
                self._lookup_table = tf.get_variable(
                    'lookup_table',
                    initializer=self._table_loader,
                    trainable=self.trainable)
                self.params.append(self._lookup_table)
                tf.initialize_variables(self.params).run(feed_dict={self._table_loader: self._init_values})
                self.reuse = True

            flow = tf.placeholder(tf.int64, [None] + self._input_shape, 'input')
            tf.add_to_collection(GraphKeys.MODEL_INPUTS, flow)
            flow = tf.nn.embedding_lookup(self._lookup_table, flow)

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, flow)
        return flow
