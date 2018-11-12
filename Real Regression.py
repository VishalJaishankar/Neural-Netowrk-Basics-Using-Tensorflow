
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf


#  # Regression with Large Dataset

# In[3]:


# create dataset with a million points 
x_data = np.linspace(0.0,10.0,1000000)


# In[4]:


# add some noise to the data set
noise = np.random.randn(len(x_data))


# In[5]:


noise


# ### y = mx + b
# 

# ### b =5

# In[6]:


y_true = (0.5 * x_data) + 5 +noise


# In[7]:


#we have created our own line with a bit of noise to make it a bit difficut for the NN
# create pandas data frame for operatons 
x_df = pd.DataFrame(data = x_data,columns=['X Data'])


# In[8]:


y_df = pd.DataFrame(data = y_true,columns=['Y'])


# In[9]:


#.head funtion gives the first 5 rows of the data frame
x_df.head()


# In[10]:


y_df.head()


# In[11]:


# pandas help with concatinating lists 
# concatinate the two lists 
# second parameter tells along which axes you want ot concat ,here it is by column
my_data = pd.concat([x_df,y_df],axis=1)


# In[12]:


my_data


# In[13]:


# plotting this whole thing would crash the kernel so pandas help in sampling and plotting it
# this function returns random 250 pair of points
my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')


# In[14]:


# you cant feed a millions points at once so you feed in batches od data


# In[15]:


np.random.randn(2)


# In[16]:


batch_size = 8 #you pass 8 samples at a time
# initialize with some random value m,b
m = tf.Variable(-0.26)
b = tf.Variable(1.37)


# In[17]:


# create placeholders
# input place holder is how many samples are you passing ,here it is batch size
xph = tf.placeholder(tf.float32,[batch_size])


# In[18]:


yph = tf.placeholder(tf.float32,[batch_size])


# In[19]:


y_model = m * xph + b # this is your model


# In[20]:


# loss function
error = tf.reduce_sum(tf.square(yph-y_model))


# In[21]:


# time for optimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


# In[22]:


init =tf.global_variables_initializer()


# In[23]:


# this is used to create a random set of 8 points from the whole data set
rand_int = np.random.randint(len(x_data),size=batch_size)
rand_int
x_data[rand_int]
y_true[rand_int]


# In[24]:


with tf.Session() as sess:
    
    sess.run(init)
    # can play along with this
    batches = 2000
    
    for i in range(batches):
        
        rand_int = np.random.randint(len(x_data),size=batch_size)
       
        feed={xph:x_data[rand_int],yph:y_true[rand_int]}
    
        sess.run(train,feed_dict=feed)
    
    model_m ,model_b =sess.run([m,b])


# In[25]:


model_m


# In[26]:


model_b


# In[27]:


y_hat = x_data * model_m +model_b


# In[28]:


my_data.sample(250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(x_data,y_hat,'r')


# # This ends the simple regression part
# # There is an estimator API
# # basically use them for Classification and Regression 
# # Steps to do so:
# ### Define a list of Features Colums
# ### Create the estimator Model
# ### Create a Data Input Function
# ### Call train,evaluate and Predict methods on the estimator object

# ##### Create a Feature column

# In[29]:


# this model has only one feature which consists of numeric data
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]


# #### Here you create an estimator  
# 
# estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# In[30]:


estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


# eval is same as test cases
x_train , x_eval , y_train , y_eval = train_test_split(x_data,y_true,test_size=0.3,random_state=101)
# usuall train is 70% of the whole data


# In[33]:


print(x_train.shape)


# In[34]:


x_eval.shape


# In[35]:


# setup estimator inputs this is important
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=None,shuffle=True)


# In[36]:


# 2 more input function for evaluation Shuffle is false cause using this 
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=8,num_epochs=1000,shuffle=False)


# In[37]:


eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=8,num_epochs=1000,shuffle=False)


# In[38]:


estimator.train(input_fn=input_func,steps=1000)


# In[39]:


# get the metrics on our training data


# In[40]:


# input is the one which is not shuffled 
train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)


# In[41]:


eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)


# ##### compare training data and test data metrics got in the previous steps

# In[42]:


print('Training Data Metrics')
print(train_metrics)


# In[43]:


print('Test Data Metrics')
print(eval_metrics)


# In[44]:


brand_new_data = np.linspace(0,10,10)


# In[45]:


input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},shuffle=False)


# In[46]:


list(estimator.predict(input_fn_predict))


# In[47]:


predictions = []

for pred in estimator.predict(input_fn_predict):
    predictions.append(pred['predictions'])


# In[48]:


predictions


# In[49]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(brand_new_data,predictions,'r*')

