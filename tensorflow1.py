# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:04:23 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32) #tensorflow中很多数据是float32
y_data = x_data*0.1+0.3 #weight 0.1, bias 0.3

#create tensorflow structure start 
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))

Bias = tf.Variable(tf.zeros([1]))

y = Weights*x_data + Bias

losses = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(losses)

init = tf.global_variables_initializer()
#create tensorflow structure end
sess = tf.Session()
sess.run(init)  #激活initial

plt.figure()
for step in range(201):
    sess.run(train)
    if step%20==0:
        print (step, sess.run(Weights), sess.run(Bias))
        print (sess.run(losses))
        plt.scatter(sess.run(Weights),sess.run(Bias))
        plt.pause(0.2)
        plt.show()
        
