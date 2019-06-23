# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:16:00 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

tf.reset_default_graph()
tf.set_random_seed(1)
np.random.seed(1)

x_t = np.linspace(-7,10,2000)[:,np.newaxis]
np.random.shuffle(x_t)
noise = np.random.normal(0,2,2000)[:,np.newaxis]
y_t = np.square(x_t)+5.+noise
train_data = np.hstack((x_t,y_t))
#plt.figure()
#plt.scatter(x_t,y_t)
#plt.show()
x_test = np.linspace(-7,10,200)[:,np.newaxis]
noise1 = np.random.normal(0,2,x_test.shape)
y_test = np.square(x_test)+5.+noise1


input_size = 1
output_size = 1
ACTIVATION = tf.nn.tanh
LAYER = 7
BATCH_SIZE = 64
LR = 0.05
'''学习效率调小不加bn也不怎么会学死，加上BN之后，学习效率可以变大'''
EPOCH = 20
x = tf.placeholder('float', [None, input_size])
y = tf.placeholder('float', [None, output_size])  #[batch_size,1]
tf_is_train = tf.placeholder(tf.bool, None)
B_INIT = tf.constant_initializer(-0.2)
class BN(object):
    
    def __init__(self, batch_normalization = False):
        self.is_bn = batch_normalization
        self.pre_ac = [x]   #pre_activation
        self.winit = tf.random_normal_initializer(mean = 0.0, stddev = 0.1)
        if self.is_bn:
            self.layers_in = [tf.layers.batch_normalization(x,training = tf_is_train)] #layers_in
        else:
            self.layers_in = [x]
        for i in range(LAYER):
            self.layers_in.append(self.add_layer(self.layers_in[-1],10,ac = ACTIVATION))
        self.output = tf.layers.dense(self.layers_in[-1],1,kernel_initializer=self.winit, bias_initializer=B_INIT)
        self.loss = tf.losses.mean_squared_error(labels = y, predictions = self.output)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)
#    @staticmethod  #这里需要用到self属性，所以函数里面需要self,静态变量不适合
    def add_layer(self,x,outputsize,ac = None):
        x = tf.layers.dense(x,outputsize,kernel_initializer=self.winit, bias_initializer=B_INIT)
        self.pre_ac.append(x)
        if self.is_bn:
            x = tf.layers.batch_normalization(x,momentum = 0.4,training = tf_is_train)
        if ac is None:
            out = x
        else:
            out = ac(x)
        return out
    
nets = [BN(batch_normalization = False), BN(batch_normalization = True)]

### plot
fig, ax = plt.subplots(4,LAYER+1, figsize=(12,6))
ax[2,0].set_ylabel(r'$Layer\ in act$')
ax[3,0].set_ylabel(r'$Layer\ BN act$')
ax[0,0].set_ylabel(r'$Pre$')
ax[1,0].set_ylabel(r'$Pre\ BN$')
for m in range(LAYER+1):
    ax[0,m].set_title('Layer%s'%m)
    for j in range(4):
        ax[j,m].set_yticks(())
    ax[0,m].set_xticks((-4,4))
    ax[1,m].set_xticks((-4,4))
#plt.show()
plt.ion()

def plot_histogram(l_in,l_in_bn,pre,pre_bn,epoch):
    for i,(l_in,l_in_bn,pre,pre_bn) in enumerate(zip(l_in,l_in_bn,pre,pre_bn)):
        if i == 0:
            p_range = (-7,10)
            l_range = (-7,10)
        else:
            p_range = (-4,4)
            l_range = (-1,1)
        ax[0,i].hist(pre.ravel(), color = '#FF9359', bins = 10, range = p_range)
        ax[1,i].hist(pre_bn.ravel(), color = '#74BCFF', bins = 10, range = p_range)  
        ax[2,i].hist(l_in.ravel(),color = '#FF9359', bins = 10, range = l_range)
        ax[3,i].hist(l_in_bn.ravel(), color = '#74BCFF',bins = 10, range = l_range)
#        plt.text(30,30,'EPOCH: %s'%epoch)
    plt.pause(0.02)
    
#data_train
sess = tf.Session()
sess.run(tf.global_variables_initializer())
loss_epoch = [[],[]]
for epoch in range(EPOCH+1):
    print ('EPOCH:', epoch)
        
    step = 0
    in_epoch = True
    while in_epoch:
        b_s,b_f = (step*BATCH_SIZE) % len(train_data),((step+1)*BATCH_SIZE) % len(train_data)
        step+=1
        if b_f< b_s:
            in_epoch = False
            b_f = len(train_data)
        x_train = train_data[b_s:b_f,0:1] #注意a[:,0:1]和a[:,0]的区别，前者还是二维的一个列向量，后者是一维
        y_train = train_data[b_s:b_f,1:2] #[batch_size,1]     
        sess.run([nets[0].train_op,nets[1].train_op],
                 feed_dict = {x:x_train,y:y_train,tf_is_train:True})
        if step ==10:
            l_in_20,l_in_bn_20,pre_20,pre_bn_20,losses,losses_bn = sess.run(
                    [nets[0].layers_in,nets[1].layers_in,nets[0].pre_ac,
                      nets[1].pre_ac,nets[0].loss,nets[1].loss],
                     feed_dict = {x:x_test,y:y_test,tf_is_train:False})
            [loss.append(l) for loss,l in zip(loss_epoch,[losses,losses_bn])]
            plot_histogram(l_in_20,l_in_bn_20,pre_20,pre_bn_20,epoch)

plt.ioff()

plt.figure(2)
plt.plot(loss_epoch[0], c='#FF9359',linewidth = 3, label = 'Without BN')
plt.plot(loss_epoch[1], c='#74BCFF',linewidth = 3, label = 'BN')
plt.legend(loc = 'best')
plt.ylim((0, 2000))
plt.title('LOSS')

y_pre, y_pre_bn = sess.run([nets[0].output, nets[1].output], 
                           feed_dict = {x:x_test,y:y_test,tf_is_train:False})  
'''train 和 test阶段在batch normalization要分开是因为计算均值和方差在train过程中会有比例(momentum)的继承,
在test阶段直接计算总样本即可，为了缩减计算量，直接代入计算公式得出BN之后的值，因为train中需要normalization
之后再加上\gamma和\beta，而测试阶段直接带入式子算出'''
plt.figure(3)
plt.scatter(x_test,y_test,c = 'blue',s = 30)
plt.plot(x_test,y_pre, c = '#FF9359',label = 'Without BN',linewidth = 5)
plt.plot(x_test,y_pre_bn, c = '#74BCFF',label = 'With BN',linewidth = 5)
plt.legend(loc = 'best')
plt.title('PREDICTION')
plt.show()



