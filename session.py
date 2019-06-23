# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:07:34 2019

@author: 12718
"""
import tensorflow as tf

matrix1 = tf.constant([[3,4]])
matrix2 = tf.constant([[1],
                       [2]])
product = tf.matmul(matrix1,matrix2)

##create the tensorflow structure end

#method 1
sess = tf.Session()
print (sess.run(product))
sess.close()
#method 2
with tf.Session() as sess:   #with A as B,将A打开然后赋值给B 
    print (sess.run(product))   #让会话层去运行结构中的product结构
    



