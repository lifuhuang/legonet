# -*- coding: utf-8 -*-
"""
Created on Sat May 14 12:20:30 2016

@author: lifu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:35:12 2016

@author: lifu
"""

import numpy as np
from legonet.models import NeuralNetwork
from legonet.layers import Input, FullyConnected, Convolution2D, Pooling2D
from legonet.optimizers import Adam

from tensorflow.examples.tutorials.mnist import input_data

###### Set mode to 'train' or 'test'#####
mode = 'train'
#########################################


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

nn = NeuralNetwork(optimizer=Adam(), log_dir='logs',
                   loss_fn='softmax_cross_entropy', target_dtype=tf.float32)
nn.add(Input('input', [28, 28, 1], ))
nn.add(Convolution2D(name='conv1', filter_height=3, filter_width=3, 
                     n_output_channels=512, activation_fn='relu'))
nn.add(Pooling2D('pooling1'))
nn.add(Convolution2D(name='conv3', filter_height=3, filter_width=3, 
                     n_output_channels=256, activation_fn='relu'))
nn.add(FullyConnected('output', 10))
nn.build()


X_train = np.array(map(lambda im: im.reshape(28, 28, 1), mnist.train.images))
Y_train = mnist.train.labels
X_test = np.array(map(lambda im: im.reshape(28, 28, 1), mnist.test.images))[:1000]
Y_test = mnist.test.labels[:1000]
try:
    nn.load_checkpoint('./checkpoints/')
    print 'checkpoint loaded!'
except Exception as e:
    print 'File not found!'
    
if mode == 'train':
    nn.fit(X_train, Y_train, n_epochs=10, batch_size=64,
           freq_checkpoint=100 , freq_log=10, 
           checkpoint_dir='./checkpoints/', loss_decay=0.9)
elif mode == 'test':
    Y_pred = nn.predict(X_test)
    print 'accuracy', (np.sum(np.argmax(Y_pred, axis=1) == 
        np.argmax(Y_test, axis=1))) * 100.0 / X_test.shape[0] 