# -*- coding: utf-8 -*-
"""
Created on Sun May 12 14:13:47 2019

@author: 12718
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

with tf.variable_scope('scope1') as scope:
    initializer = tf.constant_initializer(0.1)
    a1 = tf.get_variable('a1',shape = [1],initializer = initializer)
    scope.reuse_variables()
    a2 = tf.get_variable(name = 'a1')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(a1))
    print (sess.run(a2))
    
    

with tf.name_scope('name_scope') as scope:
#    initializer = tf.constant_initializer(0.2)
    a1 = tf.Variable(tf.constant(0.1,shape = [1]),name = 'a1')
    a2 = tf.Variable(name = 'a1',initial_value = [1])
    a3 = tf.get_variable(name = 'a1',shape = [1],initializer = tf.constant_initializer(0.1))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (a1.name)
    print (a2.name)
    print (a3.name)
    