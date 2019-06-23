# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:22:12 2019

@author: 12718
"""
'''tensorflow中使用变量都需要全局初始化，并且需要sess.run(init)'''
import tensorflow as tf

state = tf.Variable(0,name = 'counter')
print (state.name)

cons1 = tf.constant(1)

new_value = tf.add(state,cons1)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        sess.run(update)
        print (sess.run(state))