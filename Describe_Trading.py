#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# read the trading data
trade = pd.read_csv('NCCU_SEC_PROFIT.csv')
trade


# In[3]:


# read the inventory data
inventory = pd.read_csv('NCCU_SEC_MARKET_VALUE.csv')
inventory


# In[4]:


# Data Duration
print('(Trade)Begin: ', min(trade['TRADE_DATE']),'End: ', max(trade['TRADE_DATE']))
print('(Inventory)Begin: ', min(inventory['DATA_DATE']),'End: ', max(inventory['DATA_DATE']))


# In[5]:


# Numbers of sample
print('(Trade)Numbers of users: ', trade['ACCT_KEY'].nunique())
print('(Inventory)Numbers of users: ', inventory['ACCT_KEY'].nunique())


# In[6]:


# Characteristics
samples = pd.DataFrame({'ACCT_KEY':trade['ACCT_KEY'].unique()})
trait = trade[['ACCT_KEY','GENDER_CODE','AGE','JOB_DESC']]
samples = pd.merge(samples, trait, how='left',on='ACCT_KEY')
samples = samples.drop_duplicates(subset=['ACCT_KEY']).sort_values(by='ACCT_KEY')
samples = samples.set_index('ACCT_KEY')
samples


# In[7]:


samples.to_csv('samples_trait.csv')


# In[8]:


# Gender Counts
male = samples[samples.GENDER_CODE==1]
female = samples[samples.GENDER_CODE==2]
print('Male: ',male['GENDER_CODE'].count())
print('Female: ', female['GENDER_CODE'].count())


# In[9]:


# List of jobs and their population
jobs = trade[['ACCT_KEY','JOB_DESC']]
jobs = jobs.drop_duplicates(subset=['ACCT_KEY']).sort_values(by='ACCT_KEY')
jobs = jobs.groupby('JOB_DESC').count()
jobs


# In[10]:


jobs = pd.DataFrame({'JOB_DESC':samples['JOB_DESC'].unique()})
for each in jobs['JOB_DESC']:
    cnt = samples[samples.JOB_DESC==each]['JOB_DESC'].count()
    print(each, cnt)


# In[11]:


# Total numbers of trading record
counts = trade.count()['TRADE_DATE']
print('Total numbers of trading record: ', counts)


# In[12]:


# Buy/Secll Counts
buy_counts = trade[trade.BUY_SELL=='B'].count()['TRADE_DATE']
sell_counts = trade[trade.BUY_SELL=='S'].count()['TRADE_DATE']
print('Buys: ',buy_counts,'Sells: ',sell_counts)


# In[13]:


# Trading frequency
freq = pd.DataFrame(trade.groupby('ACCT_KEY').count()['TRADE_DATE'])
freq = freq.rename({'TRADE_DATE':'T_FREQ'}, axis='columns')
freq


# In[14]:


# Sum of profit/loss and TXN_AMOUNT
profit = trade.groupby('ACCT_KEY').sum()[['TXN_AMOUNT','PROFIT']]
profit = pd.DataFrame(profit)
profit


# In[15]:


FxP = pd.merge(freq, profit, on='ACCT_KEY')
FxP = pd.merge(samples, FxP, on='ACCT_KEY')
FxP


# In[16]:


# Weighted Beta
inventory['P_BETA21'] = inventory['PERCENTAGE'] * inventory['BETA_21']
inventory['P_BETA65'] = inventory['PERCENTAGE'] * inventory['BETA_65']
inventory['P_BETA250'] = inventory['PERCENTAGE'] * inventory['BETA_250']
inventory


# In[17]:


# Portfolio Beta for each record
beta = inventory.groupby(['DATA_DATE','ACCT_KEY']).sum()[['P_BETA21','P_BETA65','P_BETA250']]
beta


# In[18]:


# Mean portfolio beta for each user
mean_beta = beta.groupby('ACCT_KEY').mean()[['P_BETA21','P_BETA65','P_BETA250']]
mean_beta


# In[19]:


FxBxP = pd.merge(FxP, mean_beta, on='ACCT_KEY')
FxBxP


# In[20]:


FxBxP.to_csv('Trading_Inventory_Summary.csv')


# In[21]:


# Histogram for the profit of each transaction
x = trade['PROFIT']
plt.hist(x, bins=10)
plt.gca().set(title='PROFIT Histogram', ylabel='PROFIT')


# In[23]:


trade['PROFIT'].plot.kde()


# In[24]:


# Histogram for the profit of each transaction by different gender
trade['PROFIT'].hist(by=trade['GENDER_CODE'])


# In[26]:


# Histogram for the profit of each transaction by different job
trade['PROFIT'].hist(by=trade['JOB_DESC'])


# In[ ]:





# In[28]:




