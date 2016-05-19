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
from legonet.layers import Input, FullyConnected, Convolution, Pooling
from legonet.optimizers import Adam

from tensorflow.examples.tutorials.mnist import input_data

###### Set mode to 'train' or 'test'#####
mode = 'train'
#########################################


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

nn = NeuralNetwork(
    Adam(), 'logs', loss_fn='softmax_cross_entropy', target_dtype='float32')
nn.add(Input([28, 28, 1]))
nn.add(Convolution([3, 3], 256))
nn.add(Convolution([3, 3], 512))
nn.add(Pooling())
nn.add(FullyConnected(256, 'relu'))
nn.add(FullyConnected(10))
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
    nn.fit(X_train, Y_train, n_epochs=10, batch_size=16,
           freq_checkpoint=100 , freq_log=10, 
           checkpoint_dir='./checkpoints/', loss_decay=0.9)
elif mode == 'test':
    Y_pred = nn.predict(X_test)
    print 'accuracy', (np.sum(np.argmax(Y_pred, axis=1) == 
        np.argmax(Y_test, axis=1))) * 100.0 / X_test.shape[0] 