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


# read the data
behavior = pd.read_csv('NCCU_BEHAVIOR.csv').sort_values(by='TIME').reset_index(drop=True)
behavior


# In[3]:


# the list of the cards 
cards = pd.DataFrame({'CARD_NAME':behavior['CARD_NAME'].unique()})
cards


# In[4]:


len(cards)


# In[5]:


# the list of the cards and their popularity of interaction
card_name = pd.DataFrame(behavior.groupby(["ACTION","CATEGORY","CARD_NAME"]).count()['TIME'])
card_name


# In[6]:


# sort
card_sort = card_name['TIME'].groupby(level=0, group_keys=False)
card_sort = pd.DataFrame(card_sort.nlargest(len(cards)))
card_sort


# In[7]:


card_sort.to_csv('card_name.csv')


# In[8]:


# merge symbol_id and raw columns as they are mutually exclusive but actually the same thing
behavior.SYMBOL_ID.fillna(behavior.RAW, inplace=True)
del behavior['RAW']


# In[9]:


# the list of assets and their popularity of interaction
symbol = pd.DataFrame(behavior.groupby('SYMBOL_ID').count()['DATE'])
symbol = symbol.rename({'DATE':'TIMES'},axis='columns').sort_values(by='TIMES',ascending=False)
symbol.head(10)


# In[10]:


# create a new column to count
behavior['B_FREQ']=0
behavior


# In[11]:


# count the frequency of behavior
# set value equals 1 if ACCT_KEY or action change
for i in range(0,len(behavior)-1):
    if behavior.iloc[i]['ACCT_KEY'] != behavior.iloc[i+1]['ACCT_KEY']     or behavior.iloc[i]['ACTION'] != behavior.iloc[i+1]['ACTION']:
        behavior.at[i,'B_FREQ'] = 1
    else:
        continue
behavior


# In[12]:


# examine
behavior.to_csv('behavior.csv')


# In[13]:


# frequency table by each account
behavior_freq = pd.DataFrame(behavior.groupby('ACCT_KEY').sum())
behavior_freq


# In[14]:


behavior_freq.to_csv('Behavior_Summary.csv')


# In[ ]:




