# -*- coding: utf-8 -*-
"""
Created on Sun May  5 16:08:11 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
time0 = time.time()
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

x = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float', [None, 10])  # 0~9 digits

def conv2(x, w):
    return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = 'SAME')

def pooling_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME')

### convolution layer1 
w_conv1 = tf.Variable(tf.truncated_normal([5,5,1,8], stddev = 0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape = [8]))
x_in = tf.reshape(x, [-1, 28,28, 1])
h_conv1 = tf.nn.relu(conv2(x_in,w_conv1)+b_conv1) #[-1,28,28,1] ==> [-1,28,28,8]
h_pool1 = pooling_2x2(h_conv1) #[-1,14,14,8]

### convolution layer2
w_conv2 = tf.Variable(tf.truncated_normal([5,5,8,16],stddev = 0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape = [16]))
h_conv2 = tf.nn.relu(conv2(h_pool1, w_conv2)+b_conv2) #[-1,14,14,8] ==>[-1,14,14,16]
h_pool2 = pooling_2x2(h_conv2) #[-1,7,7,16]

### fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.layers.dense(h_pool2_flat, 512))

### dropout layyer ==> reduce the overfitting, always use at fully connected layer
keep_prob = tf.placeholder('float') 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### output layer
w_fc2 = tf.Variable(tf.truncated_normal([512, 10], stddev = 0.1))
b_fc2 = tf.Variable(tf.constant(0.1,shape = [10]))
h_fc2 = tf.matmul(h_fc1_drop, w_fc2)+b_fc2

### cross_entropy
#cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = h_fc2, labels = y_)) #1
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels = y_, weights = 1.0, logits = h_fc2)
'''上面的方法已经对交叉熵求平均'''
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(tf.nn.softmax(h_fc2),1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

### train
batch_size = 100
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(3000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict = {y_:batch_y, x:batch_x, keep_prob:0.5})
        if i%500 == 0:
            train_accuracy = sess.run(accuracy, feed_dict = {y_:batch_y, x:batch_x, keep_prob:1.0})
            print ('step %4d, | the train accuracy:'%i, train_accuracy)
            
    test_accuracy = sess.run(accuracy, feed_dict = {y_:mnist.test.labels, x:mnist.test.images, keep_prob:1.0})

print ('the test accuracy:', test_accuracy)
time1 = time.time()
print ('the time of calculation: ', time1-time0)




