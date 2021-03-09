#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


T_I = pd.read_csv('Trading_Inventory_Summary.csv')
B = pd.read_csv('Behavior_Summary.csv')


# In[3]:


T_I


# In[4]:


B


# In[5]:


df = pd.merge(T_I, B, how='left', on='ACCT_KEY')
df


# In[20]:


describe = df.describe()[['B_FREQ','T_FREQ','P_BETA21','P_BETA65','P_BETA250','TXN_AMOUNT','PROFIT']]
describe


# In[21]:


describe.to_csv('Describe.csv')


# In[6]:


# Histogram for the total profit of each user
x = df['PROFIT']
plt.hist(x, bins=10)
plt.gca().set(title='PROFIT Histogram', ylabel='PROFIT')


# In[7]:


# Histogram for the total profit of each user by different gender
df['PROFIT'].hist(by=df['GENDER_CODE'])


# In[8]:


# Histogram for the total profit of each user by different job
df['PROFIT'].hist(by=df['JOB_DESC'])


# In[ ]:


# Kernel Density Estimation for profit (by different Beta)


# In[9]:


# Histogram for the behavior frequence of each user
x = df['B_FREQ']
plt.hist(x, bins=10)
plt.gca().set(title='Behavior frequence Histogram', ylabel='B_FREQ')


# In[10]:


# Histogram for the behavior frequence of each user by different gender
df['B_FREQ'].hist(by=df['GENDER_CODE'])


# In[11]:


# Histogram for the behavior frequence of each user by different job
df['B_FREQ'].hist(by=df['JOB_DESC'])


# In[13]:


from sklearn import linear_model
import statsmodels.api as sm
from scipy import stats

x = np.array(df['B_FREQ']).reshape(-1,1)
y = df['PROFIT']

X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[14]:


x = pd.concat([df['B_FREQ'],df['P_BETA250']],axis=1)
y = df['PROFIT']

X2 = sm.add_constant(x)
model = sm.OLS(y, X2)
model2 = model.fit()
print(model2.summary())


# In[ ]:




