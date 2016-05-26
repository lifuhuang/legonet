# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:35:12 2016

@author: lifu
"""

import numpy as np
from legonet.models import NeuralNetwork
from legonet.layers import FullyConnected, Input
from legonet.regularizers import l2
from legonet.optimizers import Adam

nn = NeuralNetwork(optimizer=Adam(), log_dir='logs')
nn.add(Input(128))
nn.add(FullyConnected(64, 'relu', weight_regularizer=l2(0.001)))
nn.add(FullyConnected(32, 'relu', weight_regularizer=l2(0.001)))
nn.add(FullyConnected(5, weight_regularizer=l2(0.001)))
nn.build()


X = np.random.randn(1000, 128)
y = np.random.randint(0, 5, 1000)

try:
    nn.load_checkpoint('./checkpoints/')
    print 'checkpoint loaded!'
except ValueError as e:
    print 'File not found!'
nn.fit(X, y, n_epochs=1000, batch_size=64, 
       freq_checkpoint=10000, checkpoint_dir='./checkpoints/', loss_decay=0.9)
