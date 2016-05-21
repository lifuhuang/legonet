# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:22:39 2016

@author: lifu
"""

import numpy as np
import tensorflow as tf

from legonet.models import NeuralNetwork
from legonet.layers import Input, Convolution, Pooling, FullyConnected
from legonet.optimizers import Adam
from legonet.initializers import *

###### Set mode to 'train' or 'test'#####
mode = 'train'
train_path = '/mnt/shared/cifar-10/data_batch_1'
test_path = '/mnt/shared/cifar-10/test_batch'


#########################################


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary


def get_image(im):
    return im.reshape(3, 32, 32).transpose([1, 2, 0])


training_set = unpickle(train_path)
test_set = unpickle(test_path)

X_train = np.array(map(get_image, training_set['data']))
Y_train = np.array(training_set['labels'])
X_test = np.array(map(get_image, test_set['data']))[:1000]
Y_test = np.array(test_set['labels'])[:1000]

print 'Data loaded!'

nn = NeuralNetwork(optimizer=Adam(), log_dir='logs')
nn.add(Input([32, 32, 3]))
nn.add(Convolution([3, 3], 512))
nn.add(Convolution([3, 3], 256))
nn.add(Pooling())
nn.add(Convolution([3, 3], 256))
nn.add(Convolution([3, 3], 128))
nn.add(Pooling())
nn.add(Convolution([3, 3], 128))
nn.add(Convolution([3, 3], 64))
nn.add(Pooling())
nn.add(FullyConnected(32, activation_fn='relu', weight_init=truncated_normal(), bias_init=constant(0.1)))
nn.add(FullyConnected(10, weight_init=truncated_normal(), bias_init=constant(0.1), name='output'))
nn.build()

try:
    nn.load_checkpoint('./checkpoints/')
    print 'checkpoint loaded!'
except Exception as e:
    print 'File not found!'

if mode == 'train':
    nn.fit(X_train, Y_train, n_epochs=1000, batch_size=8, freq_checkpoint=100, freq_log=10,
           checkpoint_dir='./checkpoints/', loss_decay=0.9)
elif mode == 'test':
    Y_pred = nn.predict(X_test)
    print 'accuracy', (np.sum(np.argmax(Y_pred, axis=-1) ==Y_test)) * 100.0 / X_test.shape[0]
