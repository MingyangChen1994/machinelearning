# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:08:15 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lr = 0.1

init_param = [2.0,5.0]
train_param = [[4,2],
               [0.5,4.0],
               [3.4,0.1]][0]

x = np.linspace(-1,1,200)
noise = np.random.normal(0,1,200)*0.2
#y_tar = lambda a,b: a*x+b
#y_train = lambda a,b:a*x+b
y_tar = lambda a,b:np.sin(b*np.cos(a*x))
y_train = lambda a,b:tf.sin(b*tf.cos(a*x))
y_tar0 = y_tar(*init_param)+noise
#plt.scatter(x,y_tar0)
#plt.show()

a, b = [tf.Variable(initial_value = p, dtype = tf.float32) for p in train_param]
y_train0 = y_train(*[a,b])

loss_v = tf.losses.mean_squared_error(labels = y_tar0, predictions = y_train0)
train_re = tf.train.AdamOptimizer(lr).minimize(loss_v)

w_pre1 = []
b_pre1 = []
loss_pre = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50+1):
        _, loss_o,a_pre,b_pre = sess.run([train_re, loss_v,a,b])
        if i%5==0:
            w_pre1.append(a_pre)
            b_pre1.append(b_pre)
            loss_pre.append(loss_o)
            print ('loss:%s, a:%s, b:%s'%(loss_o,a_pre,b_pre))
            
### visualization of the gradient
fig = plt.figure()
#ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax = fig.gca(projection = '3d')
#ax = Axes3D(fig)
w0 = np.linspace(-2,6,100)
b0 = np.linspace(0,10,100)
w,b = np.meshgrid(w0,b0)
loss_plot = np.array([np.mean(np.square(y_tar(*[w1,b1])-y_tar0)) for w1,b1 in zip(w.flatten(),b.flatten())]).reshape(w.shape)
ax.plot_surface(w,b,loss_plot,cmap = 'rainbow',alpha = 0.5)
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.scatter(w_pre1[0],b_pre1[0],loss_pre[0], s = 30, c = 'r')
ax.plot(w_pre1,b_pre1,zs = loss_pre,zdir = 'z',c = 'r',linewidth = 3)
plt.show()

            
            
            
