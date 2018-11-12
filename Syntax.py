
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np


# In[10]:


a = tf.constant(10)
b = tf.constant(20)


# In[11]:


with tf.Session() as sess:
    
    result=sess.run(a+b)


# In[12]:


result


# In[13]:


fill_mat = tf.fill((4,4),10)
zero = tf.zeros((4,4))
ones = tf.ones((4,4))
distu = tf.random_uniform((4,4),minval=0,maxval=10)
distn = tf.random_normal((4,4),mean=0,stddev=1)


# In[14]:


#now try running this in a session
my_matrices=[fill_mat,zero,ones,distn,distu]


# In[17]:


with tf.Session() as sess:
    result=sess.run(my_matrices)


# In[18]:


result


# In[19]:


sess=tf.InteractiveSession()


# In[20]:


for op in my_matrices:
    print(sess.run(op))


# In[22]:


a =tf.constant([[1,2],[3,4]])


# In[25]:


sess.run(a)


# In[27]:


a.get_shape()


# In[28]:


b = tf.constant([[10],[100]])
b.get_shape()


# In[29]:


#now try matrix multiplication
result = tf.matmul(a,b)


# In[31]:


sess.run(result)


# In[32]:


result.eval()

