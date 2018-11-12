
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[4]:


#placeholders are initially empty and will be used to store the data
#it needs to know the type and the size


# In[5]:


sess = tf.InteractiveSession()


# In[6]:


my_tensor = tf.random_uniform((4,4),minval=0,maxval=1)


# In[17]:


my_tensor


# In[18]:


my_var = tf.Variable(initial_value=my_tensor)


# In[19]:


print(my_var)


# In[20]:


#sess.run(my_var)# have to initialize ,otherwise you cannot use it


# In[21]:


# after creating all the variables you run the init to do the initialization
init = tf.global_variables_initializer()


# In[22]:


#this does the initialization
sess.run(init)


# In[23]:


#use variables after initialization
sess.run(my_var)


# In[24]:


#creating placeholders
ph = tf.placeholder(tf.float32)

