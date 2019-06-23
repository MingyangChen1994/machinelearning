# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:48:59 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

print (mnist.train.images.shape)
print (mnist.train.labels.shape)
plt.imshow(mnist.train.images[0].reshape([28,28]), cmap = 'gray')
plt.title('%i'%np.argmax(mnist.train.labels[0]))
plt.show()

x = tf.placeholder("float32", [None, 784])
y_ = tf.placeholder("float32", [None, 10])



#### 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)
#### 权重初始化end
    
#### 卷积和池化
def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME') #[1,stride_x, stride_y,1]

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'SAME')
#ksize = [1, patch_x, patch_y, 1]
#####################
    
### The first convolution and pooling 
W_conv1 = weight_variable([5,5,1,8])  #[patch_w, patch_h, in_channel_, out_channel]  
b_conv1 = bias_variable([8])

x_image = tf.reshape(x, [-1, 28, 28, 1]) #-1 recept any number of samples

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1) #[-1,28,28,1]-->[-1,28,28,8]
h_pool1 = max_pool_2x2(h_conv1) #[-1,28,28,8]-->[-1,14,14,8]

### The second convolution and pooling 
W_conv2 = weight_variable([5,5,8,16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2) #[-1,14,14,16]
h_pool2 = max_pool_2x2(h_conv2) #[-1,7,7,16]

### The fully connected 
W_fc1 = weight_variable([7*7*16, 512])
b_fc1 = bias_variable([512])
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

### Dropout--> to reduce the overfitting 
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### Output layer
W_fc2 = weight_variable([512, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

### Training and assessment
'''不指定axis直接计算所有的和'''
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.))) #clip_by_value避免log(0)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver({'W_out':W_fc2})
###saver读取时，变量名字需要和存取一样才能找到，并且第二次读取的时候会给设置的待存放变量加上_2这种，使其出错
for i in range(10000+1):
    batch_x, batch_y = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict = {keep_prob:0.5, x:batch_x, y_:batch_y})
    if i%1000 == 0:
        with sess.as_default():
            train_accuracy = accuracy.eval(feed_dict = {keep_prob:1.0, x:batch_x, y_:batch_y})
            print ("step%d, training accuracy: %g" %(i, train_accuracy))
            saver.save(sess, 'saver\\cnn.ckpt', global_step = i)
            
with sess.as_default():
    print ("test accuracy: %g",
           accuracy.eval(feed_dict = {keep_prob:1.0, x:mnist.test.images, y_:mnist.test.labels}))




