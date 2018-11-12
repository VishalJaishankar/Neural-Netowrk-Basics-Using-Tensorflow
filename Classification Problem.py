
# coding: utf-8

# In[ ]:


# classification Problem
import pandas as pd
import numpy as np


# In[2]:


diabetes = pd.read_csv('diabetes.csv')
diabetes.shape


# In[3]:


diabetes.head()


# In[4]:


diabetes.columns


# In[5]:


# choose columns to normalize
cols_to_norm = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction']


# In[6]:


# this function apply is used for normalization 
diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x-x.min()) / (x.max()-x.min() ))


# In[7]:


diabetes.head()


# In[8]:


# create a feature colums


# In[9]:


import tensorflow as tf


# In[10]:


diabetes.columns


# In[11]:


# create a new feature 
preg = tf.feature_column.numeric_column('Pregnancies')
glu = tf.feature_column.numeric_column('Glucose')
press = tf.feature_column.numeric_column('BloodPressure')
skin = tf.feature_column.numeric_column('SkinThickness')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('DiabetesPedigreeFunction')
age = tf.feature_column.numeric_column('Age')


# In[12]:


# dealing with noncontinuous values
#converting continuous column into catagorical column
#feature engineering


# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


diabetes['Age'].hist(bins=20)


# In[15]:


#to bucket a continuous value
age_bucket = tf.feature_column.bucketized_column(age,boundaries=[20,30,40,50,60,70,80])


# In[16]:


#now declare the feature columns as defined before by tf
feat_cols = [preg,glu,press,skin,insulin,bmi,diabetes_pedigree,age_bucket]


# #### Now perform train test split
# 

# In[17]:


# This removes the result column
x_data = diabetes.drop('Outcome',axis=1)


# In[18]:


x_data.head()


# In[19]:


label = diabetes['Outcome']


# In[20]:


label.head()


# In[21]:


# now Perform the train test split


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(x_data,label,test_size=0.3,random_state=101)


# ### Create the input Funtion

# In[24]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


# In[25]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)


# In[26]:


#now train the model


# In[27]:


model.train(input_fn=input_func,steps=1000)


# In[28]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)


# In[29]:


results = model.evaluate(eval_input_func)


# In[30]:


results


# #### get some predictions
# 

# In[31]:


pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)


# In[32]:


my_pred = model.predict(pred_input_func)


# In[33]:


for item in my_pred:
    print(item)


# ### Doing a Dense NN Classifier

# In[34]:


# hidden units provide how many layers you want and how many neurons are there per layer
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)


# In[35]:


dnn_model.train(input_fn = input_func,steps=1000)


# In[36]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)


# In[37]:


dnn_model.evaluate(eval_input_func)

