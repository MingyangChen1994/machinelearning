# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:30:55 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()  #避免出现kernel exists问题
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

### parameters 
lr = 0.001
train_iters = 1200
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder('float', [None, n_steps, n_inputs])
y_ = tf.placeholder('float', [None, n_classes])


## RNN
def RNN(x):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = 100)
    # x==> [batch_size, 28steps, 28inputs]
    #outputs ==> [batch_size, 28steps, num_units]
    outputs, states = tf.nn.dynamic_rnn(lstm_cell,x,time_major = False, dtype = tf.float32)
    #fully connected layer, here 10 is the output size of this layer
    '''
    tf.layers.dense(

    inputs,
 
    units, 输出的维度

    activation=None,

    use_bias=True,

    kernel_initializer=None,  ##卷积核的初始化器

    bias_initializer=tf.zeros_initializer(),  ##偏置项的初始化器，默认初始化为0

    kernel_regularizer=None,    ##卷积核的正则化，可选

    bias_regularizer=None,    ##偏置项的正则化，可选

    activity_regularizer=None,   ##输出的正则化函数

    kernel_constraint=None,   

    bias_constraint=None,

    trainable=True,

    name=None,  ##层的名字

    reuse=None  ##是否重复使用参数

)

    '''
    output = tf.layers.dense(outputs[:,-1,:], 10)  #outputs [batch_size, 28, 100]
    return output

output = RNN(x)
y_pre = tf.nn.softmax(output)


### cross_entropy
#tf.nn.softmax_cross_entropy_with_logits   return an arrays
cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y_))

correct_pre = tf.equal(tf.argmax(y_, 1), tf.argmax(y_pre, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, dtype = tf.float32))

### The optimizer method
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# train 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_iters+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_inputs))
        sess.run(train_step, feed_dict = {y_:batch_y, x:batch_x})
        if i%50 ==0:
            train_accuracy = sess.run(accuracy, feed_dict = {y_:batch_y, x:batch_x})
            print ('training accuracy:', train_accuracy)
       
    test_accuracy = sess.run(accuracy, feed_dict = {y_:mnist.train.labels, x:mnist.train.images})
    
print ('test accuracy:', test_accuracy)
    
    


















