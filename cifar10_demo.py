# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:22:39 2016

@author: lifu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 14 12:20:30 2016

@author: lifu
"""


import numpy as np
import tensorflow as tf

from legonet.models import NeuralNetwork
from legonet.layers import Input, Output, Convolution2D, Pooling2D
from legonet.optimizers import Adam


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

training_set = unpickle(train_path)
test_set = unpickle(test_path)

X_train = np.array(map(lambda im: im.reshape(32, 32, 3), training_set['data']))
Y_train = np.array(training_set['labels'])
X_test = np.array(map(lambda im: im.reshape(32, 32, 3), test_set['data']))
Y_test = np.array(test_set['labels'])

nn = NeuralNetwork(optimizer=Adam(0.01), log_dir='logs')
nn.add(Input('input', [32, 32, 3], ))
nn.add(Convolution2D(name='conv1', filter_height=3, filter_width=3, 
                     n_output_channels=256, activation_fn='relu'))                     
nn.add(Convolution2D(name='conv2', filter_height=3, filter_width=3, 
                     n_output_channels=128, activation_fn='relu'))
nn.add(Pooling2D('pooling1', mode='max'))
nn.add(Convolution2D(name='conv3', filter_height=3, filter_width=3, 
                     n_output_channels=64, activation_fn='relu'))
nn.add(Convolution2D(name='conv4', filter_height=3, filter_width=3, 
                     n_output_channels=32, activation_fn='relu'))
nn.add(Output(loss_fn='sparse_softmax_cross_entropy', output_fn='softmax',
              name='output', target_shape=[], target_dtype=tf.int64, 
              output_shape=10))
nn.build()


try:
    #nn.load_checkpoint('./checkpoints/')
    print 'checkpoint loaded!'
except Exception as e:
    print 'File not found!'
    
if mode == 'train':
    nn.fit(X_train, Y_train, n_epochs=10, batch_size=64,
           freq_checkpoint=20 , freq_compute_loss=1, 
           checkpoint_dir='./checkpoints/', loss_decay=0.9)
elif mode == 'test':
    Y_pred = nn.predict(X_test)
    print 'accuracy', (np.sum(np.argmax(Y_pred, axis=1) == 
        np.argmax(Y_test, axis=1))) * 100.0 / X_test.shape[0] 