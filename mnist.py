# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:14:37 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#读取数据集，以0:[1000000000]的形式
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
xs = tf.placeholder(tf.float32, [None, 784]) #imgage
ys = tf.placeholder(tf.float32, [None,10])  #label

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]))
    Wx_plus_b = tf.matmul(inputs, Weights)+biases
    if activation_function is None:
        return Wx_plus_b
    else:
        return activation_function(Wx_plus_b)
    
def accuracy(v_x, v_y):
    global y
    
    correct_prediction = tf.equal(tf.argmax(v_y,1), tf.argmax(y,1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return sess.run(accuracy1, feed_dict = {xs:v_x})

y = add_layer(xs, 784, 10, activation_function = tf.nn.softmax) #softmax = normalize(exp(y))
cross_entropy = -tf.reduce_sum(ys*tf.log(y)) #计算所有元素的和
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)



init = tf.global_variables_initializer()
sess =  tf.Session() 
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100) #随机梯度
    sess.run(train_step, feed_dict = {xs:batch_xs, ys:batch_ys})
    if i%50 ==0:
        print (accuracy(mnist.test.images, mnist.test.labels))

      
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(ys,1))
#accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print (sess.run(accuracy1, feed_dict={xs: mnist.test.images, ys: mnist.test.labels}))





