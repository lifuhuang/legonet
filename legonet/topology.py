# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:12:10 2016

@author: lifu
"""

import tensorflow as tf

class Node(object):
    """Base class for all kinds of layers.
    """
    
    # TODO: add operators: +, &
    
    def __init__(self, name):
        """Initialize a new Node instance.
        
        Since Node is a abstract class, this method should only be called by
        its derived classes.
        """
        
        self.name = name
        
        self.input = None
        self.output = None
        
        self.pred = None
        self.succ = None

    def connect_to(self, predecessor):
        """Connect this Node to another layer.
        """
        
        self.pred = predecessor
        if predecessor is not None:
            self.pred.succ = self
    
    def build(self):
        """Construct the Node in tensorflow graph.
        """
        
        raise NotImplementedError
        
class Sequential(Node):
    """# TODO: docstring
    """
    
    
    def __init__(self, name):
        """Initialize a new instance of Sequential.
        """        
        
        super(Sequential, self).__init__(name)
        
        self.layers = []
    
    def build(self):
        """Construct the Sequential and its layers.
        """
        
        # build graph at layer level
        with tf.variable_scope(self.name):
            for i, layer in enumerate(self.layers): 
                if i == 0:
                    layer.connect_to(self.pred)
                else:
                    layer.connect_to(self.layers[i-1])
                layer.build()
            
        # keep record of input/ouput of model
        self.input = self.layers[0].input
        self.output = self.layers[-1].output  

    def add(self, layer):
        """Add a layer to this network.
        """
        
        self.layers.append(layer)

class Parallel(Node):
    """# TODO: docstring
    """
    
    
    def __init__(self, name, mode='concat', along_dim=None):
        """Initialize a new instance of Parallel.
        """        
        
        if mode == 'concat' and along_dim is None:
            raise ValueError('Must specify along_dim for concat mode.')
        super(Parallel, self).__init__(name)
        
        self.layers = []
        if mode not in ['concat', 'sum', 'mean']:
            raise ValueError('Unknown mode: {0}'.format(mode))            
        self.mode = mode
        self.along_dim = along_dim
        
    def build(self):
        """Construct the Sequential and its layers.
        """
        
        # build graph at layer level
        with tf.variable_scope(self.name):
            for i, layer in enumerate(self.layers): 
                layer.connect_to(self.pred)
                layer.build()
                
            # keep record of input/ouput of model
            self.input = self.pred.output
            outputs = list(l.output for l in self.layers)
            if self.mode == 'concat':
                self.output = tf.concat(self.along_dim, outputs)
            elif self.mode == 'mean':
                self.output = tf.add_n(outputs) / len(self.layers)
            elif self.mode == 'sum':
                self.output = tf.add_n(outputs)

    def add(self, layer):
        """Add a layer to this network.
        """
        
        self.layers.append(layer)
        