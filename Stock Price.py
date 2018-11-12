
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[353]:


milk = pd.read_csv('Monthly2.csv',index_col='Month')


# In[354]:


milk.shape


# In[355]:


milk.index = pd.to_datetime(milk.index)


# In[356]:


milk.plot()


# In[357]:


milk.info()


# In[358]:


train_set = milk.head(156)


# In[359]:


test_set = milk.tail(12)


# In[360]:


from sklearn.preprocessing import MinMaxScaler


# In[361]:


scaler = MinMaxScaler()


# In[362]:


train_scaled = scaler.fit_transform(train_set)


# In[363]:


test_scaled = scaler.transform(test_set)


# ### Creating a Batch Function

# In[364]:


def next_batch(training_data,batch_size,steps):
    """
    INPUT: Data,batch
    """
    rand_start = np.random.randint(0,len(training_data)-steps)
    
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    
    
    return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1)


# ### Setup the RNN Model

# In[365]:


import tensorflow as tf
tf.reset_default_graph()


# ##### Defining Constants

# In[366]:


num_inputs = 1

num_time_steps = 12

num_neurons = 100

num_outputs = 1

learning_rate = 0.01

num_train_interations = 8000

batch_size = 1


# #### Create PlaceHolders

# In[367]:


X = tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])


# In[368]:


cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons,activation=tf.nn.relu)
        ,output_size=num_outputs)


# ##### Pass in the cell variables and the input placeholder(X)

# In[369]:


outputs , state =tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)


# ##### Create the Loss Function and Optimizer

# In[370]:


loss = tf.reduce_mean(tf.square(outputs-y))


# In[371]:


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)


# ##### Init Global Variable

# In[372]:


init = tf.global_variables_initializer()


# #### Create instance of tf.train.Saver()

# In[373]:


saver = tf.train.Saver()


# #### Running Session 

# ##### To avoid no memory error allocate GPU memory for faster computation.

# In[374]:


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


# In[375]:


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    sess.run(init)
                
    for iteration in range(num_train_interations):
                
        X_batch , y_batch = next_batch(train_scaled,batch_size,num_time_steps)
            
        sess.run(train,feed_dict={X:X_batch,y:y_batch})
                
        if iteration%100 == 0:
                
            mse = loss.eval(feed_dict={X:X_batch,y:y_batch})
            print(iteration,"\tMSE",mse)
            
    # this saves the model and can be reloaded when needed later
    saver.save(sess,"./time_series_model")


# ##### Predictions for Last 12 Months

# In[376]:


test_set


# In[377]:


with tf.Session() as sess:
     
    saver.restore(sess,"./time_series_model")
    
    train_seed = list(train_scaled[-12:])
    
    for iteration in range(12):
        
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1,num_time_steps,1)
        
        y_pred = sess.run(outputs,feed_dict={X:X_batch})
        
        train_seed.append(y_pred[0,-1,0])
        
        
                          


# In[378]:


train_seed


# In[379]:


result = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))


# In[380]:


test_set['Generated']=result


# In[381]:


test_set


# In[382]:


test_set.plot()

