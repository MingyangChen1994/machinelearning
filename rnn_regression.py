# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:00:15 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

# construct two sets of numbers(y1, y2), and learn from y1 to predict y2

# show data
step0 = 0
x_ = np.linspace(step0*np.pi, (step0+2)*np.pi, 100, dtype = np.float32)
y_label0 = np.sin(x_)
y_input0 = np.cos(x_)
#plt.figure()
#plt.plot(x_, y_label0,'r-', label = 'target(sin)')
#plt.plot(x_, y_input0,'g',label = 'input(cos)')
#plt.legend(loc = 'best')

time_steps = 20
input_size = 1
batch_size = 1
cell_size = 64
lr = 0.005

x = tf.placeholder('float', [None, time_steps, input_size],name = 'x_in')
y = tf.placeholder('float', [None, time_steps, input_size],name = 'y_tar')

###  RNN layer
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)
initial = lstm_cell.zero_state(batch_size = batch_size, dtype = tf.float32)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, time_major = False, initial_state = initial)
## outputs size [1, time_steps, cell_size]
output2D = tf.reshape(outputs, [-1,cell_size])
output = tf.layers.dense(output2D, input_size) #[1*time_steps, cell_size] ==> [1*time_steps, input_size]
output = tf.reshape(output, [-1,time_steps, input_size],name = 'output') #[1, time_steps, input_size]

## cost
loss = tf.losses.mean_squared_error(labels = y, predictions = output)
tf.add_to_collection('loss',loss)  #tensor 类型的 list  #生成一个loss名字的集合，将loss添加进去
train_op = tf.train.AdamOptimizer(lr).minimize(loss)



## train
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(max_to_keep = 1)
#plt.figure(num = 1, figsize = (12,8))
#plt.ion()
for steps in range(0,200+2,2):
    step = np.linspace(steps*np.pi, (steps+2)*np.pi, time_steps)
    x_train = np.cos(step)[np.newaxis,:,np.newaxis]
    y_train = np.sin(step)[np.newaxis,:,np.newaxis]
    if 'final_s_' not in globals():
        feed_dict = {x:x_train, y:y_train}
    else:
        feed_dict = {x:x_train, y:y_train, initial:final_s_}
    sess.run(train_op, feed_dict = feed_dict)
    final_s_ = sess.run(states, feed_dict = feed_dict)
    y_pre = sess.run(output, feed_dict = feed_dict)
    if steps%50 ==0:
        print (saver.save(sess, 'saver\\rnn_regeression.ckpt',write_meta_graph=False))
#    print (steps)
'''得到可训练的变量集合'''
#tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)  #1
#tf.trainable_variables()    #2
#tf.add_to_collection(xxx)   


#def load_test():
#saver = tf.train.import_meta_graph('saver\\rnn_regeression.ckpt.meta')
#g = tf.get_default_graph()
#with tf.Session() as sess:
#    saver.restore(sess, 'saver\\rnn_regeression.ckpt')
#    m = np.linspace(0,np.pi*2,20)
#    y_input = np.cos(m)[np.newaxis,:,np.newaxis]
#    y_target = np.sin(m)[np.newaxis,:,np.newaxis]
#    y_pre = g.get_tensor_by_name('output:0')
#    loss1 = g.get_collection('loss')[0]  #collection操作使用和tensor区别
#    x = g.get_tensor_by_name('x_in:0')
#    y = g.get_tensor_by_name('y_tar:0')
#    y_predic = sess.run(y_pre,feed_dict = {x:y_input})
#    loss = sess.run(loss1, feed_dict = {x:y_input, y:y_target})
#    print ('the loss of the test: ',loss)
#    plt.plot(m,np.sin(m),'r--',label = 'target')
#    plt.plot(m,y_predic.flatten(),'g',label = 'prediction')
#    plt.legend(loc = 'best')
#load_test()
















