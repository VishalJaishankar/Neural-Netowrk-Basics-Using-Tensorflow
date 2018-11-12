import tensorflow as tf
#   to ignore the warning for GPU processing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)
#   tensor - n dimentional array
#   basic tensor
#   constant
#
hello=tf.constant("Hello ")
world=tf.constant("World")
#   note that hello is not a string but a tensor object

#   run these things in a session

with tf.Session() as sess:
    #   everything here is tensorflow operation
    result=sess.run(hello+world)

#print(result)
a=tf.constant(10)
b=tf.constant(20)

with tf.Session() as sess:
    result=sess.run(a+b)
#print(result)

# numpy operations are possible in tf
fill_mat=tf.fill((4,4),10)

my_zeros=tf.zeros((4,4))

my_ones=tf.ones((4,4))
#   this creates a random normal distribution with the specified mean and standard deviation
myrandn = tf.random_normal((4,4),mean=0,stddev=1.0)
#   many such distributions can be created ...here i am creating uniform distribution
myrandu = tf.random_uniform((4,4),minval=1,maxval=50)

#  cant run anythin here

my_ops=[fill_mat,my_ones,my_zeros,myrandn,myrandu]

#create a

