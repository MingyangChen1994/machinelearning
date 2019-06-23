# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:33:31 2019

@author: 12718
"""

import tensorflow as tf
'''placeholder可以从外部输入变量'''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

sumnum = tf.add(input1,input2)
with tf.Session() as sess:
    print (sess.run(sumnum,feed_dict={input1:1,input2:2}))
    