# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:22:39 2016

@author: lifu
"""

import numpy as np
import tensorflow as tf

from legonet.models import NeuralNetwork
from legonet.layers import Input, Output, Convolution2D, Pooling2D, FullyConnected
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
    dict = cPickle.load(fo)
    fo.close()
    return dict

def get_image(im):
    return im.reshape(3, 32, 32).transpose([1, 2, 0])
    
training_set = unpickle(train_path)
test_set = unpickle(test_path)

X_train = np.array(map(get_image, training_set['data']))
Y_train = np.array(training_set['labels'])
X_test = np.array(map(get_image, test_set['data']))[:1000]
Y_test = np.array(test_set['labels'])[:1000]

print 'Data loaded!'

nn = NeuralNetwork(optimizer=Adam(0.01), log_dir='logs')
nn.add(Input('input', [32, 32, 3], ))
nn.add(Convolution2D(name='conv1', filter_height=3, filter_width=3, 
                     n_output_channels=256, activation_fn='relu'))
nn.add(Pooling2D('pooling1', mode='max', pool_shape=(2, 2)))
nn.add(Convolution2D(name='conv2', filter_height=3, filter_width=3, 
                     n_output_channels=256, activation_fn='relu'))
nn.add(Pooling2D('pooling2', mode='max', pool_shape=(2, 2)))
nn.add(Convolution2D(name='conv3', filter_height=3, filter_width=3, 
                     n_output_channels=256, activation_fn='relu'))
nn.add(Pooling2D('pooling3', mode='max', pool_shape=(2, 2)))
nn.add(FullyConnected('fc1', 384, activation_fn='relu',
                      weight_init=truncated_normal(), bias_init=constant(0.1)))
nn.add(FullyConnected('fc2', 192, activation_fn='relu', 
                      weight_init=truncated_normal(), bias_init=constant(0.1)))
nn.add(Output(loss_fn='sparse_softmax_cross_entropy', output_fn='softmax',
              name='output', target_shape=[], target_dtype=tf.int64, 
              output_shape=10))
nn.build()


try:
    nn.load_checkpoint('./checkpoints/')
    print 'checkpoint loaded!'
except Exception as e:
    print 'File not found!'
    
if mode == 'train':
    nn.fit(X_train, Y_train, n_epochs=1000, batch_size=32,
           freq_checkpoint=100 , freq_log=10, 
           checkpoint_dir='./checkpoints/', loss_decay=0.9)
elif mode == 'test':
    Y_pred = nn.predict(X_test)
    print 'accuracy', (np.sum(np.argmax(Y_pred, axis=-1) == 
        Y_test)) * 100.0 / X_test.shape[0] 
