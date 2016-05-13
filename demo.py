# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:35:12 2016

@author: lifu
"""

import numpy as np
import tensorflow as tf
from legonet.models import NeuralNetwork
from legonet.layers import FullyConnected, Input, Output
from legonet.regularizers import l2
from legonet.optimizers import Adam

nn = NeuralNetwork(optimizer=Adam(), log_dir='logs')
nn.add(Input(50, 'input', dtype=tf.float32))
nn.add(FullyConnected(256, 'hidden1', 'relu', weight_reg=l2(0.01)))
nn.add(FullyConnected(128, 'hidden2', 'relu', weight_reg=l2(0.01)))
nn.add(FullyConnected(64, 'hidden3', 'relu', weight_reg=l2(0.01)))
nn.add(Output(loss_fn='sparse_softmax_cross_entropy', output_fn='softmax',
              name='output', output_shape=5, target_shape=[1], 
              dtype=tf.int64, weight_reg=l2(0.01)))
nn.build()


X = np.random.randn(1000, 50)
y = np.random.randint(0, 2, (1000, 1))

try:
    nn.load_checkpoint('./checkpoints/')
    print 'checkpoint loaded!'
except Exception as e:
    print 'File not found!'
nn.fit(X, y, n_epochs=10, batch_size=64,
       freq_checkpoint=10000, checkpoint_dir='./checkpoints/', loss_decay=0.9)
