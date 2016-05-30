# -*- coding: utf-8 -*-
"""
This module contains the abstract base class `Piece` and other topological structures that play the roles of containers
or layouts in LegoNet. These classes are useful for building complicated non-sequential models, and will enable users to
build these non-sequential models in the same way as how they build sequential ones(simply calling `add` for times).

An alternative to using these topological structures when building non-sequential models is to directly use layers
in functional style(e.g. ``FullyConnected(128)(input_tensor)``). This method is even more powerful and enables users
to build more complicated models, but requires more careful design and coding.

"""


import abc

import tensorflow as tf


class Piece(object):
    """Abstract base class for all elements in graph.

    Attributes:
        name: The name of this `Piece`, might be used for visualization. Use default if `None` is passed.
        reuse: Indicates whether or not this `Piece` is in reuse mode.

    """

    # TODO: add operators: +, &

    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None):
        """Initializes a new `Piece` instance.
        
        Since `Piece` is a abstract class, this method should only be called by
        its derived classes.

        Args:
            name: Name of this `Piece`. Use default name if `None` is passed.

        """

        self.name = name
        self.reuse = False

    @abc.abstractmethod
    def __call__(self, flow):
        """Constructs this `Piece` in a `TensorFlow` graph.

        Args:
            flow: The input `Tensor` to this `Piece`.

        Returns:
            The output `Tensor` of this `Piece`.

        """

        pass


class Plate(Piece):
    """Abstract base class for all kinds of layouts.

    Plate is a special kind of `Piece`, which itself contains other `Piece`s (including other `Plate` s). Different
    `Plate` s might use different strategies for how its inner `Piece` s are connected, but looking from the outside,
    `Plate` acts totally the same as other `Piece` s. They are very useful for constructing complicated non-sequential
    neural network models.

    Attributes:
        child_pieces: a list of `Piece` s that are contained within a `Plate` instance.

    """

    def __init__(self, name=None):
        """Initializes a new `Plate` instance.

        Args:
            name: Name of this `Plate`. Use default name if `None` is passed.
        """

        super(Plate, self).__init__(name)
        self.child_pieces = []

    def add(self, piece):
        """Adds a `Piece` to this `Plate`.

        Args:
            piece: A `Piece` object.

        Returns:
            None

        """

        self.child_pieces.append(piece)


class Sequential(Plate):
    """A `Plate` whose inner `Piece` s are in a sequential layout."""

    def __init__(self, name=None):
        """Initializes a new `Sequential` instance.

        Args:
            name: Name of this `Piece`. Use default name if `None` is passed.

        """

        super(Sequential, self).__init__(name)

        self.child_pieces = []

    def __call__(self, flow=None):
        """Constructs the Sequential and its inner `Piece` s.

        Args:
            flow: Input `Tensor` object. (Default value = None)

        Returns:
            Output of this `Sequential`.

        """

        with tf.variable_op_scope([flow], self.name, 'Sequential', reuse=self.reuse):
            if not self.reuse:
                self.reuse = True
            for piece in self.child_pieces:
                flow = piece(flow)

        return flow


class Parallel(Plate):
    """A `Plate` whose inner pieces are in a parallel layout."""

    def __init__(self, name=None, mode='concat', along_dim=None):
        """Initializes a new instance of Parallel.

        Args:
            name: Name of this `Piece`. Use default name if `None` is passed.
            mode: The way to merge paralleled `Piece` s. Now supports "concat", "sum", "mean".
            along_dim: The dimension along which the merging operation will be done. Only take effect in "concat"
        mode.

        """

        if mode == 'concat' and along_dim is None:
            raise ValueError('Must specify along_dim for concat mode.')
        super(Parallel, self).__init__(name)

        self.child_pieces = []
        if mode not in ['concat', 'sum', 'mean']:
            raise ValueError('Unknown mode: {0}'.format(mode))
        self.mode = mode
        self.along_dim = along_dim

    def __call__(self, flow=None):
        """Constructs the Sequential and its inner pieces.

        Args:
            flow: Input `Tensor` object. (Default value = None)

        Returns:
            Output of this `Parallel`.

        """

        # build inner pieces.
        with tf.variable_op_scope([], self.name, 'Parallel', reuse=self.reuse):
            if not self.reuse:
                self.reuse = True

            outputs = []
            for i, piece in enumerate(self.child_pieces):
                outputs.append(piece(flow))

            if self.mode == 'concat':
                return tf.concat(self.along_dim, outputs)
            elif self.mode == 'mean':
                return tf.add_n(outputs) / len(self.child_pieces)
            elif self.mode == 'sum':
                return tf.add_n(outputs)
