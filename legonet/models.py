# -*- coding: utf-8 -*-
"""
This module contains different models, which are the interface for training and using neural networks.
"""


import os.path

import numpy as np
import tensorflow as tf

from . import activations
from . import objectives
from . import GraphKeys
from .topology import Sequential


class NeuralNetwork(object):
    """Base classes of all neural networks."""

    def __init__(self, optimizer, log_dir=None, output_fn="softmax", loss_fn="sparse_softmax_cross_entropy",
                 target_dtype=tf.int64, topology=None, graph=None, session=None):
        """Initializes a new instance of NeuralNetwork.

        Args:
            optimizer: The optimizer used when training.
            log_dir: The path of folder to output log files. Will not save log files if `None` is passed.
            output_fn: A `str` or `callable` that indicates the function imposed on the output of `model`. `softmax`
        is a common choice. Do not impose any output function to output if `None` is passed.
            loss_fn: A 'str', `callable`. Uses `sparse_softmax_cross_entropy` as default if `None` is passed.
            target_dtype: The data type of targets.
            topology: A `Node` object representing the topological structure of this neural network in the highest
        level. Will generate a new `Sequential` object as default if `None` is passed.
            graph: A TensorFlow `Graph`, will create a new one if `None` is passed.
            session: A TensorFlow `Session`, will create a new one if `None` is passed.

        """

        if topology is None:
            topology = Sequential('core')

        self._topology = topology
        self._optimizer = optimizer
        self._log_dir = log_dir

        self._target_dtype = target_dtype

        # functions
        if isinstance(output_fn, str):
            self._output_fn = activations.get(output_fn)
        elif callable(output_fn):
            self._output_fn = output_fn
        else:
            raise ValueError("output_fn should be either a str or callable")

        if isinstance(loss_fn, str):
            self._loss_fn = objectives.get(loss_fn)
        elif callable(loss_fn):
            self._loss_fn = loss_fn
        else:
            raise ValueError("loss_fn should be either a str or callable")

        # placeholders
        self._input = None
        self._targets = None

        # tensors        
        self._output = None
        self._unregularized_loss = None
        self._loss = None
        self._global_step = None
        self._update_op = None
        self._merged_summaries = None
        self._saver = None

        self._graph = graph if graph else tf.Graph()
        self._session = session if session else tf.Session(graph=self._graph)

        self._built = False

    def __del__(self):
        """Destructor of NeuralNetwork."""

        self._session.close()

    def load_checkpoint(self, path=None):
        """Loads checkpoint from a file or directory.

        Args:
            path: Path of folder containing checkpoint files. (Default value = None)

        Returns:
            None

        """

        if os.path.isdir(path):
            path = tf.train.latest_checkpoint(path)
        self._saver.restore(self._session, path)

    def fit(self, x, y, n_epochs=5, batch_size=32, checkpoint_dir=None, randomized=True, freq_log=100,
            freq_checkpoint=10000, loss_decay=0.0):
        """Trains this model using x and y.

        Args:
            x: Input array.
            y: Targets array, should be consistent with target_dtype.
            n_epochs: Number of epochs to iterate. (Default value = 5)
            batch_size: Size of mini-batch. (Default value = 32)
            checkpoint_dir: Path to the folder to store checkpoint files. (Default value = None)
            randomized: Indicates whether mini-batches will be selected randomly. (Default value = True)
            freq_log: The frequency of logging. (Default value = 100)
            freq_checkpoint: The frequency of saving parameters to checkpoint files. (Default value = 10000)
            loss_decay: The decay rate used for displaying exponential moving average of loss. (Default value = 0.0)

        Returns:
            None

        """

        assert x.shape[0] == y.shape[0]

        if not self._built:
            raise ValueError("Model has not been built.")

        if self._log_dir is None:
            sw_train = None
        else:
            sw_train = tf.train.SummaryWriter(
                os.path.join(self._log_dir, 'train'))

        epoch_size = x.shape[0]
        step = self._session.run(self._global_step)
        ema_loss = None
        try:
            for _ in xrange(n_epochs * epoch_size):
                if randomized:
                    batch_indices = np.random.randint(0, epoch_size, batch_size)
                else:
                    bg = step * batch_size % epoch_size
                    ed = (bg + batch_size) % epoch_size
                    batch_indices = np.arange(bg, ed)

                x_batch = x[batch_indices]
                y_batch = y[batch_indices]

                # update
                fetches = [self._update_op, self._global_step]
                feed_dict = {self._input: x_batch,
                             self._targets: y_batch}

                _, step = self._session.run(fetches, feed_dict=feed_dict)
                if step % freq_log == 0:
                    fetches = [self._merged_summaries, self._loss]
                    summary, batch_loss = self._session.run(fetches, feed_dict)

                    msg = 'Step: {0}, training loss: {1}'.format(step, batch_loss)
                    if loss_decay == 0:
                        print msg
                    else:
                        if ema_loss is None:
                            ema_loss = batch_loss
                        else:
                            ema_loss -= ((1.0 - loss_decay) * (ema_loss - batch_loss))
                        print msg, '(ema:{0})'.format(ema_loss)

                    if sw_train is not None:
                        sw_train.add_summary(summary, step)

                if checkpoint_dir is not None and step % freq_checkpoint == 0:
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    checkpoint_path = os.path.join(
                        checkpoint_dir, 'checkpoint')
                    cp_path = self._saver.save(self._session, checkpoint_path, step)
                    print 'A checkpoint has been saved to {0}'.format(cp_path)
        except KeyboardInterrupt:
            print 'Training process terminated by keyboard interrupt.'
        finally:
            if sw_train is not None:
                sw_train.close()

    def predict(self, x):
        """Outputs result given input.

        Args:
            x: Input array.

        Returns:
            None

        """

        return self._session.run(self._output, feed_dict={self._input: x})

    def build(self):
        """Constructs the whole neural network in tensorflow graph."""

        with self._graph.as_default():
            with self._session.as_default():
                # TODO: create collection of outputs
                raw_output = self._topology()

                all_vars_before = set(tf.all_variables())
                # keep record of input/output of model
                self._input = tf.get_collection(GraphKeys.MODEL_INPUTS)[0]
                self._output = self._output_fn(raw_output)
                self._targets = tf.placeholder(self._target_dtype, name='target')
                self._unregularized_loss = self._loss_fn(raw_output, self._targets)

                # build graph at network level
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                self._loss = tf.add_n(
                    [self._unregularized_loss] + reg_losses, name='loss')

                self._global_step = tf.get_variable(
                    'global_step', [], tf.int64,
                    initializer=tf.zeros_initializer,
                    trainable=False)

                self._update_op = self._optimizer.minimize(self._loss, global_step=self._global_step)

                # summaries
                for activation in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
                    tf.histogram_summary(activation.name, activation)
                    tf.scalar_summary('{0} sparsity'.format(activation.name), tf.nn.zero_fraction(activation))
                tf.scalar_summary("Loss", self._loss)
                self._merged_summaries = tf.merge_all_summaries()
                all_vars_after = set(tf.all_variables())
                tf.initialize_variables(all_vars_after - all_vars_before).run()
                self._saver = tf.train.Saver()
                self._graph.finalize()

        if self._log_dir is not None:
            sw = tf.train.SummaryWriter(self._log_dir, graph=self._graph)
            print 'Graph visualization has been saved to {0}'.format(self._log_dir)
            sw.close()

        self._built = True

    def add(self, layer):
        """Adds a layer to the model inside this NeuralNetwork.

        Args:
            layer: a `Layer` instance.

        Returns:
            None

        """

        self._topology.add(layer)
