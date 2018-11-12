
# coding: utf-8

# In[243]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[244]:


price = pd.read_csv('^GSPC.csv',index_col='Date')


# In[245]:


price['Mid']=(price['High']+price['Low'])/2
price.head()


# In[246]:


price = price.drop(['Open','Close','Adj Close','Volume','High','Low'],axis=1)


# In[247]:


price.head()


# In[248]:


price.index = pd.to_datetime(price.index)


# In[249]:


price.plot()


# In[250]:


price.shape


# In[251]:


#split into test-train
train_set = price.head(240)
train_set


# In[252]:


test_set = price[240:252]
test_set


# In[253]:


from sklearn.preprocessing import MinMaxScaler


# In[254]:


scaler = MinMaxScaler()


# In[255]:


train_scaled = scaler.fit_transform(train_set)


# In[256]:


test_scaled = scaler.transform(test_set)


# In[257]:


def next_batch(training_data,batch_size,steps):
    """
    INPUT: Data,batch
    """
    rand_start = np.random.randint(0,len(training_data)-steps)
    
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)
    
    
    return y_batch[:,:-1].reshape(-1,steps,1) , y_batch[:,1:].reshape(-1,steps,1)


# In[258]:


import tensorflow as tf
tf.reset_default_graph()


# In[259]:


num_inputs = 1

num_time_steps = 12

num_neurons = 1000

num_outputs = 1

learning_rate = 0.0003

num_train_interations = 1000

batch_size = 1


# In[260]:


X = tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])


# In[261]:


cell = tf.contrib.rnn.OutputProjectionWrapper(
        tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons,activation=tf.nn.relu)
        ,output_size=num_outputs)


# In[262]:


outputs , state =tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)


# In[263]:


loss = tf.reduce_mean(tf.square(outputs-y))


# In[264]:


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)


# In[265]:


init = tf.global_variables_initializer()


# In[266]:


saver = tf.train.Saver()


# In[267]:


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


# In[268]:


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    
    sess.run(init)
                
    for iteration in range(num_train_interations):
                
        X_batch , y_batch = next_batch(train_scaled,batch_size,num_time_steps)
            
        sess.run(train,feed_dict={X:X_batch,y:y_batch})
                
        if iteration%100 == 0:
                
            mse = loss.eval(feed_dict={X:X_batch,y:y_batch})
            print(iteration,"\tMSE",mse)
            
    # this saves the model and can be reloaded when needed later
    saver.save(sess,"./time_series_model_new")


# In[ ]:


test_set


# In[ ]:


with tf.Session() as sess:
     
    saver.restore(sess,"./time_series_model_new")
    
    train_seed = list(train_scaled[-12:])
    
    for iteration in range(12):
        
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1,num_time_steps,1)
        
        y_pred = sess.run(outputs,feed_dict={X:X_batch})
        
        train_seed.append(y_pred[0,-1,0])


# In[ ]:


train_seed


# In[ ]:


result = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))


# In[ ]:


test_set['Generated']=result


# In[ ]:


test_set


# In[ ]:


test_set.plot()

