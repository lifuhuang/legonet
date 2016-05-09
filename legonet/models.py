import numpy as np
import tensorflow as tf


class NeuralNetwork(object):
    """Base classes of all neural networks.
    """
    
    
    def __init__(self, optimizer, graph=None, session=None):
        """Initialize a new instance of NeuralNetwork.
        """
        
        self.layers = []
        self.input = None
        self.output = None
        self.loss = None
        self.update = None
        self.optimizer = optimizer
        
        self.graph = graph if graph else tf.Graph()
        self.session = session if session else tf.Session(graph=self.graph)
        self.built = False
    
    def __del__(self):
        """Destructor of NeuralNetwork.
        """
        
        self.session.close()
        
    def fit(self, x, y, n_epochs=5, batch_size=32, 
            randomized=True, use_checkpoint=False, freq_compute_loss=100):
        """Train this model using x and y.
        """
        
        assert x.shape[0] == y.shape[0]
        
        if not self.built:
            raise ValueError("Model has not been built.")
        
        epoch_size = x.shape[0]
        
        for i in xrange(0, n_epochs * epoch_size):
            if randomized:
                batch_indices = np.random.randint(0, epoch_size, batch_size)
            else:
                bg = i * batch_size % epoch_size
                ed = (bg + batch_size) % epoch_size
                batch_indices = np.arange(bg, ed)
                
            x_batch = x[batch_indices]
            y_batch = y[batch_indices]
            
            # update
            feed_dict = {self.input: x_batch,
                         self.targets: y_batch}
            if i % freq_compute_loss:                    
                self.session.run([self.update], feed_dict=feed_dict)
            else:
                loss, _ = self.session.run([self.loss, self.update], 
                                           feed_dict=feed_dict)
                print 'Iteration {0} with training loss: {1}'.format(i, loss)
     
    def predict(self, x):
        """Output result given input.
        """
        
        return self.session([self.output], feed_dict={self.input: x})
                        
    def build(self):
        """Construct the whole neural network in tensorflow graph.
        """
        
        with self.graph.as_default():
            for i, layer in enumerate(self.layers): 
                layer.build(self.layers[i-1] if i > 0 else None)
                
            self.input = self.layers[0].output
            self.output = self.layers[-1].output        
            self.loss = self.layers[-1].loss
            self.targets = self.layers[-1].targets
            self.update = self.optimizer.minimize(self.loss)
            self.session.run(tf.initialize_all_variables())
            self.graph.finalize()
            
        self.built = True
            
    def add(self, layer):
        """Add a layer to this network.
        """
        
        self.layers.append(layer)

   