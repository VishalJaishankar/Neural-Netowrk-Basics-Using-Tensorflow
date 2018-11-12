
# coding: utf-8

# In[73]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf


# In[74]:


first = pd.read_csv('housing.csv')


# In[75]:


first.head()


# In[76]:


first.columns


# In[77]:


data1 = first[['housing_median_age','total_rooms','total_bedrooms','population','households','median_income','median_house_value']]


# In[78]:


# all the features are continuous
data1.head()


# In[79]:


from sklearn.model_selection import train_test_split 
#split the data into training and testing


# In[80]:


x_data1 = data.drop('median_house_value',axis=1)
x_data1.head()
y_data1 = data['median_house_value']
y_data1.head()


# In[81]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(x_data1,y_data1,test_size=0.3,random_state=101)


# In[82]:


X_train1.head()


# In[83]:


y_train1.head()


# In[84]:


X_test1.head()


# In[85]:


from sklearn.preprocessing import MinMaxScaler


# In[86]:


scaler = MinMaxScaler()


# In[87]:


scaler.fit(X_train1)


# In[88]:


scaler.transform(X_train1)


# In[89]:


# with this scaled X_test create a dataframe


# In[90]:


scaler.transform(X_test1)


# In[91]:


X_train1 = pd.DataFrame(scaler.transform(X_train1),columns=['housing_median_age','total_rooms','total_bedrooms','population',
                                                       'households','median_income'],index=X_train1.index)


# In[92]:


X_train1.head()


# In[93]:


X_test1 = pd.DataFrame(scaler.transform(X_test1),columns=['housing_median_age','total_rooms','total_bedrooms','population',
                                                       'households','median_income'],index=X_test.index)


# In[94]:


X_test1.head()


# ### Create Feature Columns

# In[95]:


house_age = tf.feature_column.numeric_column('housing_median_age')
rooms = tf.feature_column.numeric_column('total_rooms')
beds = tf.feature_column.numeric_column('total_bedrooms')
pop = tf.feature_column.numeric_column('population')
hold = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('median_income')


# In[96]:


feat_cols1 = [house_age,rooms,beds,pop,hold,income]


# In[97]:


input_func1 = tf.estimator.inputs.pandas_input_fn(x=X_train1,y=y_train1,batch_size=5,num_epochs=1000,shuffle=True)


# In[98]:


dnn_model1 = tf.estimator.LinearRegressor(feat_cols1)


# In[99]:


dnn_model1.train(input_fn = input_func1,steps=1000)

