
# coding: utf-8

# In[ ]:


#   NN for linear fit for a 2D data
# Steps to be followed:
# Build a GRAPH
# Init a Session
# Feed Data In and Get Output
import tensorflow as tf
import numpy as np


# In[73]:


# weight is a variable as seen before
# and the input is the placeholder
# Build a cost function to optimize parameters 'w' and 'b'


# In[74]:


# create arandom seed
np.random.seed(101)
tf.set_random_seed(101)


# In[75]:


rand_a = np.random.uniform(0,100,(5,5))


# In[76]:


rand_a


# In[77]:


rand_b = np.random.uniform(0,100,(5,1))


# In[78]:


rand_b


# In[79]:


# create placeholders for random uniform objects
a = tf.placeholder(tf.float32)


# In[80]:


b = tf.placeholder(tf.float32)


# In[81]:


# create operations ..My understanding:they act like functions these operations and when you call you assign values
add_op = a+b


# In[82]:


mult_op = a * b


# In[83]:


# you will be using a feed dictionary to assign the place holders a value
with tf.Session() as sess:
    add_result = sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
    print(add_result)
    
    mul_res = sess.run(mult_op,feed_dict={a:rand_a,b:rand_b})
    print(mul_res)


# In[84]:


# what we did was to create a session graph


# In[85]:


# our data has 10 features
n_features = 10
#define how many neurons are gonna be there in different layers
n_dense_neurons = 3


# In[86]:


# first paramenter depends on how big the batch is you feed for the NN second parameter would be the features
# Row is number of samples and column is number of features
x = tf.placeholder(tf.float32,shape=(None,n_features))


# In[87]:


# declare variables
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
# see how the sizes are determined
b = tf.Variable(tf.ones([n_dense_neurons]))


# In[88]:


# define the graphs for the sessions
xW = tf.matmul(x,W)


# In[89]:


z = tf.add(xW,b)


# In[90]:


# pass this into an activation function (this z)
a = tf.sigmoid(z)


# In[91]:


# initiliza all the variables
init = tf.global_variables_initializer()


# In[92]:


with tf.Session() as sess:
    #run the initializer 
    sess.run(init)
    # put the output as the output of the activatio function and feed the values
    layer_out = sess.run(a,feed_dict={x:np.random.random([1,n_features])})


# In[93]:


print(layer_out)


# In[94]:


# This does not achieve anything since we start with random values
# Have a cost function and do optimization and adjust W,b
## Simple Regression Example


# In[95]:


# give me 10 random number between 0 and 10 with some noise of range -1.5 to 1.5
x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)


# In[96]:


x_data


# In[97]:


y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)


# In[98]:


y_label


# In[99]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[101]:


plt.plot(x_data,y_label,'*')


# In[105]:


# the model we are trying to fit is 
# y = mx + b


# In[112]:


np.random.rand(2)


# In[113]:


# initialize variable with this value
m = tf.Variable(0.05)
b = tf.Variable(0.63)
# the neural network will work on these two and make them right to fit these


# In[114]:


# create a cost function
error = 0
for x,y in zip(x_data,y_label):
    #zip creates a tuple of the inputs
    y_hat = m*x+b #this is our predicted value
    
    error +=(y-y_hat)**2    #our minimization function


# In[115]:


# now we have to optimize this error


# In[117]:


# Tensorflow has the training thing too
# the parameters are just the learning rate start with 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#the above step basically is to declare an optimizer with the specified learning rate
# now tell the optimizer what to do i.e to minimize the error function
train = optimizer.minimize(error)


# In[118]:


# do the gloabal init for the variable 
init = tf.global_variables_initializer()


# In[127]:


with tf.Session() as sess:
    
    sess.run(init)
    
    training_steps = 1000
    
    for i in range(training_steps):
        
        sess.run(train)
    
    final_slope,final_intercept = sess.run([m,b])


# In[128]:


x_test = np.linspace(-1,11,10)
# plot the line supposidly fiven by the net runnug training_steps times
# y = mx + b
y_pred_plot = final_slope*x_test +final_intercept 

plt.plot(x_test,y_pred_plot,'r')
# take the original points as well
plt.plot(x_data,y_label,'*')

