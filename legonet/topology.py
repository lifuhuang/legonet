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
    
    def call(self, flow, reuse=False):
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
    
    def call(self, flow=None, reuse=False):
        """Construct the Sequential and its layers.
        """
            
        # build graph at layer level
        with tf.variable_scope(self.name, reuse=reuse):
            for layer in self.layers: 
                flow = layer.call(flow, reuse)
                
        return flow

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
        
    def call(self, flow=None, reuse=False):
        """Construct the Sequential and its layers.
        """
        
        # build graph at layer level
        with tf.variable_scope(self.name, reuse=reuse):
            outputs = []
            for i, layer in enumerate(self.layers):        
                outputs.append(layer.call(flow, reuse))
                
            if self.mode == 'concat':
                return tf.concat(self.along_dim, outputs)
            elif self.mode == 'mean':
                return tf.add_n(outputs) / len(self.layers)
            elif self.mode == 'sum':
                return tf.add_n(outputs)

    def add(self, layer):
        """Add a layer to this network.
        """
        
        self.layers.append(layer)
        