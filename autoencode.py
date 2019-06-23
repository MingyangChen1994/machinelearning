# -*- coding: utf-8 -*-
"""
Created on Wed May  8 19:43:19 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = False)    
x = tf.placeholder('float',[None, 28*28])


lr = 2e-3
batch_size = 100
test_size = 5
train_steps = 10000
####  encode  4layer
en1 = tf.layers.dense(x, 256, activation = tf.nn.tanh)
env2 = tf.layers.dense(en1, 128, activation = tf.nn.tanh)
env22 = tf.layers.dense(env2, 64, activation = tf.nn.tanh)
env3 = tf.layers.dense(env22, 8, activation = tf.nn.tanh)
encode = tf.layers.dense(env3, 3)  #最后的分类数字不限范围，所以不加activation

####  outcode 4layer
out1 = tf.layers.dense(encode,8,activation = tf.nn.tanh)
out22 = tf.layers.dense(out1,64, activation = tf.nn.tanh)
out2 = tf.layers.dense(out22,128,activation = tf.nn.tanh)
out3 = tf.layers.dense(out2,256,activation = tf.nn.tanh)
outcode = tf.layers.dense(out3,784,activation = tf.nn.sigmoid)  #最后重现灰度值，sigmoid合适

loss = tf.losses.mean_squared_error(labels = x, predictions = outcode)
train = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

random = np.random.randint(0,len(mnist.test.images),test_size)
x_test = mnist.test.images[random]
fig, ax = plt.subplots(2,test_size, figsize = (10,4))
plt.ion() #continious plot

for i in range(test_size):
    ax[0][i].imshow(x_test[i].reshape(28,28),cmap = 'gray')
    ax[0][i].set_xticks(())
    ax[0][i].set_yticks(())

### train and plot the trained images
for step in range(train_steps+1):
    batch_x,batch_y = mnist.train.next_batch(batch_size)
    _,encode1,loss1 = sess.run([train,encode,loss], feed_dict = {x:batch_x})
    if step%200==0: 
#        print ('steps: %s | loss: '%step, loss1)
        decode_data = sess.run(outcode, feed_dict = {x:x_test})
        for j in range(test_size):
            ax[1][j].imshow(decode_data[j].reshape(28,28))
            ax[1][j].set_xticks(())
            ax[1][j].set_yticks(())
        plt.title(r'$step:%5d, loss:%s$'%(step,loss1))
        plt.draw()
        plt.pause(0.02)
plt.ioff()

## 3D plot of the extracted features for the test data
x_test3d = mnist.test.images[:500]
y_test3d = mnist.test.labels[:500]
   
extract_feature = sess.run(encode, feed_dict = {x:x_test3d})
fig = plt.figure(num=2)
ax = Axes3D(fig)
X,Y,Z = extract_feature[:,0],extract_feature[:,1],extract_feature[:,2]

for x_,y,z,s in zip(X,Y,Z,y_test3d):
    c = cm.rainbow(int(255*s/9))
    ax.text(x_,y,z,s,backgroundcolor = c)
    ax.set_xlim(X.min(), X.max());ax.set_ylim(Y.min(), Y.max());ax.set_zlim(Z.min(), Z.max())
    plt.show()




    
