
# coding: utf-8

# # MNIST Basic Approach(Softmax)

# In[3]:


import tensorflow as tf
# now doenload the mnist dataset from tensorflow


# In[4]:


from tensorflow.examples.tutorials.mnist import input_data


# In[5]:


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[6]:


type(mnist)


# In[7]:


mnist.train.images


# In[9]:


mnist.test.num_examples


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


# The data is an array which is flattened ,you need to reshape it 
# thisimage is represented by pixels 28X28 grid 
single_image = mnist.train.images[1].reshape(28,28)


# In[17]:


plt.imshow(single_image,cmap='gist_gray')


# #### Now Creating the model

# In[18]:


# The data is normalized already


# ### Follow the steps:

# In[26]:


# PLACEHOLDER
# We have only one set of imputs i.e the image data
x = tf.placeholder(tf.float32,shape=[None,784])
# the 784 is 28X28 since the array is flattened


# In[27]:


# VARIABLES
# our weights and biases
# weight size is 784 pixels by 10 possible labels it has 
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# In[28]:


# CREATE GRAPHS OPS
y = tf.matmul(x,W) + b


# In[29]:


# lOSS fUNCTION
y_true = tf.placeholder(tf.float32,shape=[None,10])


# In[30]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                              (labels=y_true,logits=y))


# In[32]:


# oPRIMIZER
optim = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optim.minimize(cross_entropy)


# In[33]:


# CREATE SESSION
init = tf.global_variables_initializer()


# In[ ]:


with tf.Session() as sess:
    sess.run(init)
    
    for steps in range(1000):
        
        batch_x,batch_y = mnist.train.next_batch(100)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y}) 

