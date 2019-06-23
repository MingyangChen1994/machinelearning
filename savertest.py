# -*- coding: utf-8 -*-
"""
Created on Sat May  4 15:35:22 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np

W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name="weights")
b = tf.Variable([[1,2,3]],dtype=tf.float32,name="biases")
 
init = tf.global_variables_initializer()
 
saver = tf.train.Saver()
 
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,"saver\\test.ckpt")
    print("Save to path:",save_path)

def method1():
    W1 = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
    b1 = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")
    
    #### 第二遍读取的时候会在weights上面自动加上_1,_2,_3....
    
    
    # not need init step
     
    saver1 = tf.train.Saver()
    with tf.Session() as sess:
        # 提取变量
        saver1.restore(sess,"saver\\test.ckpt")
        print("weights:",sess.run(W1))
        print("biases:",sess.run(b1))
#method1()


def method2():
    saver = tf.train.import_meta_graph('saver\\test.ckpt.meta')
    graph = tf.get_default_graph() #到默认图
    with tf.Session() as sess:
        saver.restore(sess, 'saver\\test.ckpt')
        print (sess.run(graph.get_tensor_by_name('weights:0')))  # name得加上 :0
    
method2()

    
    
    
    
    
    
    
    
    
    
    
    
    
