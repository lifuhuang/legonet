# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:12:10 2016

@author: lifu
"""


import tensorflow as tf


class Node(object):
    """Base class for all elements in graph."""

    # TODO: add operators: +, &

    def __init__(self, name=None):
        """Initialize a new Node instance.
        
        Since Node is a abstract class, this method should only be called by
        its derived classes.

        Args:
          name: Name of this `Node`. Use default name if `None` is passed.

        """

        self.name = name
        self.reuse = False

    def call(self, flow):
        """Construct the Node in tensorflow graph.

        Args:
          flow: The input `Tensor` to this `Node`.

        Returns:
          None

        """

        raise NotImplementedError


class Sequential(Node):
    """Container Node whose inner nodes are in a sequential layout."""

    def __init__(self, name=None):
        """Initialize a new instance of Sequential.

        Args:
          name: Name of this `Node`. Use default name if `None` is passed.

        """

        super(Sequential, self).__init__(name)

        self.nodes = []

    def call(self, flow=None):
        """Construct the Sequential and its `Node`s.

        Args:
          flow: Input `Tensor` object. (Default value = None)

        Returns:
          Output of this `Node`.

        """

        with tf.variable_op_scope([flow], self.name, 'Sequential', reuse=self.reuse):
            if not self.reuse:
                self.reuse = True
            for node in self.nodes:
                flow = node.call(flow)

        return flow

    def add(self, node):
        """Add a node to this network.

        Args:
          node: A `Node` object.

        Returns:
          None

        """

        self.nodes.append(node)


class Parallel(Node):
    """Container Node whose inner nodes are in a parallel layout."""

    def __init__(self, name=None, mode='concat', along_dim=None):
        """Initialize a new instance of Parallel.

        Args:
          name: Name of this `Node`. Use default name if `None` is passed.
          mode: The way to merge paralleled `Node`s. Now supports "concat", "sum", "mean".
          along_dim: The dimension along which the merging operation will be done. Only take effect in "concat"
        mode.

        """

        if mode == 'concat' and along_dim is None:
            raise ValueError('Must specify along_dim for concat mode.')
        super(Parallel, self).__init__(name)

        self.nodes = []
        if mode not in ['concat', 'sum', 'mean']:
            raise ValueError('Unknown mode: {0}'.format(mode))
        self.mode = mode
        self.along_dim = along_dim

    def call(self, flow=None):
        """Construct the Sequential and its nodes.

        Args:
          flow: Input `Tensor` object. (Default value = None)

        Returns:
          Output of this `Node`.

        """

        # build graph at node level
        with tf.variable_op_scope([], self.name, 'Parallel', reuse=self.reuse):
            if not self.reuse:
                self.reuse = True

            outputs = []
            for i, node in enumerate(self.nodes):
                outputs.append(node.call(flow))

            if self.mode == 'concat':
                return tf.concat(self.along_dim, outputs)
            elif self.mode == 'mean':
                return tf.add_n(outputs) / len(self.nodes)
            elif self.mode == 'sum':
                return tf.add_n(outputs)

    def add(self, node):
        """Add a `Node` to this network.

        Args:
          node: A `Node` object.

        Returns:
          None

        """

        self.nodes.append(node)
