# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:15:02 2019

@author: 12718
"""
'''
input layer  1
hidden layer 10 (loss train_method)
output layer 1 (prediction)

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data

'''如果bias项，所有的x都只是线性回归到y'''


def add_layer(inputs, in_size, out_size, activation_func = None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.truncated_normal([in_size,out_size],stddev = 0.1))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]))+0.1
        with tf.name_scope('Wx_plus_b'):            
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        
        ## use activation function or not?
        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
            
        return outputs

x_data = np.linspace(-1,1,1000)[:,np.newaxis] #需要转置一下，因为add_layer中in_size为1
#此处并不是转置,相当于把原来的类似于list的增加了一个空间，放入二维里面的一列
noise = np.random.normal(-0.5,0.5,x_data.shape)*0.1
y_data = np.power(x_data,2) +x_data +0.5 +noise

fig = plt.figure()
left,bottom,width,height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left,bottom,width,height])
ax1.scatter(x_data,y_data)

ax2 = fig.add_axes([0.6,0.15,0.2,0.2])
plt.ion()


with tf.name_scope('input'):
        
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input') #None 不显示sample数量


#hiden layer
l1 = add_layer(xs,1,10,activation_func = tf.nn.tanh)
prediction = add_layer(l1,10,1,activation_func = None)

#method
with tf.name_scope('loss'):    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys),reduction_indices = [0])) 
'''tf.reduce_sum()为了降维，下面方法维度是[xxx]，降维之后是xxx'''
#loss = tf.reduce_mean((tf.square(prediction-ys)))  '''也可以，实际上和上面一样'''
#这里axis = 1让数组横过来？
optimizer = tf.train.GradientDescentOptimizer(0.1)  #0.1优化速率是说每次以0.1的权重改变参数
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    writer = tf.summary.FileWriter('log\\', sess.graph)
    sess.run(init)
    n = 0
    for i in range(3000):
        sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
        if i%50 == 0:
            loss1 = sess.run(loss,feed_dict = {xs:x_data,ys:y_data})
            n+=1
            try:
                ax1.lines.remove(lines[0])  #去除第一条
            except Exception:
                pass
            lines = ax1.plot(x_data,sess.run(prediction, feed_dict = {xs:x_data}),'r-',lw = 5)
            ax1.set_title(r'$loss = %s$'% loss1)
            ax2.scatter(n,loss1)
#            plt.draw()
            plt.pause(0.2)
            plt.show()
            


