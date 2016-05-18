import os.path

import numpy as np
import tensorflow as tf

from . import activations
from . import objectives
from . import GraphKeys
from .topology import Sequential

class NeuralNetwork(object):
    """Base classes of all neural networks.
    """
    
    
    def __init__(self, optimizer, log_dir=None,
                 output_fn=None, loss_fn=None, target_dtype=tf.int64,
                 model=None, graph=None, session=None):
        """Initialize a new instance of NeuralNetwork.
        """        
        
        if output_fn is None:
            output_fn = 'softmax'            
        if loss_fn is None:        
            loss_fn = 'sparse_softmax_cross_entropy'
        if model is None:
            model = Sequential('core')
        
        self.model = model        
        self.optimizer = optimizer
        self.log_dir = log_dir
        
        self.target_dtype = target_dtype
        
        # functions
        if isinstance(output_fn, str):
            self._output_fn = activations.get(output_fn)
        else:
            self._output_fn = output_fn
            
        if isinstance(loss_fn, str):
            self._loss_fn = objectives.get(loss_fn) 
        else:
            self._loss_fn = loss_fn
                
        # placeholders
        self.input = None
        self.targets = None

        # tensors        
        self.output = None
        self.unregularized_loss = None        
        self.loss = None
        self.global_step = None
        self.update_op = None
        self.merged_summaries = None
        self.saver = None
        
        self.graph = graph if graph else tf.Graph()
        self.session = session if session else tf.Session(graph=self.graph)
        
        self.built = False
    
    def __del__(self):
        """Destructor of NeuralNetwork.
        """
        
        self.session.close()
    
    def load_checkpoint(self, path=None):
        """Load checkpoint from a file or directory.
        """
        
        if os.path.isdir(path):
            path = tf.train.latest_checkpoint(path)
        self.saver.restore(self.session, path)
    
    def fit(self, x, y, n_epochs=5, batch_size=32, checkpoint_dir=None, 
            randomized=True, freq_log=100, freq_checkpoint=10000,
            loss_decay=0.0):
        """Train this model using x and y.
        """
        
        assert x.shape[0] == y.shape[0]
        
        if not self.built:
            raise ValueError("Model has not been built.")

        if self.log_dir is None:
            sw_train = None
        else:            
            sw_train = tf.train.SummaryWriter(
                os.path.join(self.log_dir, 'train'))
                    
        epoch_size = x.shape[0]
        step = self.session.run(self.global_step)
        ema_loss = None
        try:
            for _ in xrange(n_epochs * epoch_size):
                if randomized:
                    batch_indices = np.random.randint(0, epoch_size, 
                                                      batch_size)
                else:
                    bg = step * batch_size % epoch_size
                    ed = (bg + batch_size) % epoch_size
                    batch_indices = np.arange(bg, ed)
                    
                x_batch = x[batch_indices]
                y_batch = y[batch_indices]
                
                # update
                fetches = [self.update_op, self.global_step]
                feed_dict = {self.input: x_batch,
                             self.targets: y_batch}
                             
                _, step = self.session.run(fetches, feed_dict=feed_dict)
                if step % freq_log == 0:
                    fetches = [self.merged_summaries, self.loss]
                    summary, batch_loss = self.session.run(fetches, feed_dict)

                    msg = 'Step: {0}, training loss: {1}'.format(step, 
                        batch_loss)
                    if loss_decay == 0:
                        print msg
                    else:
                        if ema_loss is None:
                            ema_loss = batch_loss
                        else:
                            ema_loss -= ((1.0 - loss_decay) * 
                                (ema_loss - batch_loss))
                        print msg, '(ema:{0})'.format(ema_loss)
                            
                    if sw_train is not None:
                        sw_train.add_summary(summary, step)
                    
                if checkpoint_dir is not None and step % freq_checkpoint == 0:
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    checkpoint_path = os.path.join(
                        checkpoint_dir, 'checkpoint')
                    cp_path = self.saver.save(
                        self.session, checkpoint_path, step)
                    print 'A checkpoint has been saved to {0}'.format(cp_path)
        except KeyboardInterrupt:
            print 'Training process terminated by keyboard interrupt.'
        finally:
            if sw_train is not None:
                sw_train.close()
     
    def predict(self, x):
        """Output result given input.
        """
        
        return self.session.run(self.output, feed_dict={self.input: x})
                        
    def build(self):
        """Construct the whole neural network in tensorflow graph.
        """
        
        with self.graph.as_default():
            with self.session.as_default():
                # TODO: create collection of outputs
                raw_output = self.model.call()
                
                all_vars_before = set(tf.all_variables())
                # keep record of input/ouput of model
                self.input = tf.get_collection(GraphKeys.MODEL_INPUTS)[0]
                self.output = self._output_fn(raw_output)
                self.targets = tf.placeholder(self.target_dtype, name='target')
                self.unregularized_loss = self._loss_fn(
                    raw_output, self.targets)
                
                # build graph at network level
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                self.loss = tf.add_n(
                    [self.unregularized_loss] + reg_losses, name='loss')
                
                self.global_step = tf.get_variable(
                    'global_step', [], tf.int64, 
                    initializer=tf.zeros_initializer,
                    trainable=False)
                    
                self.update_op = self.optimizer.minimize(
                    self.loss, global_step=self.global_step)
                
                # summaries
                for activation in tf.get_collection(tf.GraphKeys.ACTIVATIONS):
                    tf.histogram_summary(activation.name, activation)
                    tf.scalar_summary(
                        '{0} sparsity'.format(activation.name), 
                        tf.nn.zero_fraction(activation))
                tf.scalar_summary("Loss", self.loss)
                self.merged_summaries = tf.merge_all_summaries()
                all_vars_after = set(tf.all_variables())
                tf.initialize_variables(all_vars_after - all_vars_before).run()
                self.saver = tf.train.Saver()
                self.graph.finalize()
        
        if self.log_dir is not None:
            sw = tf.train.SummaryWriter(self.log_dir, graph=self.graph)
            print 'Graph visualization has been saved to {0}'.format(
                self.log_dir)
            sw.close()
                
        self.built = True
    
    def add(self, layer):
        """Add a layer to the model inside this NeuralNetwork.
        """
        
        self.model.add(layer)