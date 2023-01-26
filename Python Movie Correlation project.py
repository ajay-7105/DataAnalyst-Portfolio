#!/usr/bin/env python
# coding: utf-8

# In[13]:


#Import 
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(12,8)


# In[63]:


#Reading the data
df=pd.read_csv('movies.csv')
df
df.head()


# In[64]:


df.head()


# In[67]:


df.dtypes


# In[68]:


for col in df.columns:
    prcnt_missing=np.mean(df[col].isnull())
    print('{}-{}%'.format(col,prcnt_missing))


# In[118]:


df['votes']=df['votes'].fillna(value=df['votes'].mean())
df['gross']=df['gross'].fillna(value=df['gross'].mean())
df['budget']=df['votes'].fillna(value=df['budget'].mean())
df.head()


# In[119]:


df['gross']=df['gross'].astype('int64')
df['budget']=df['budget'].astype('int64')
df['votes']=df['votes'].astype('int64')

df.head()


# In[101]:


df_sort=df.sort_values(by=['gross'],inplace=False,ascending=False)


# In[120]:


pd.set_option('display.max_rows',None)
df.head()


# In[80]:


#Correlation
#Scatter plot (Budget vs gross)

plt.scatter(x=df['budget'],y=df['gross'])
plt.title('budget vs gross-earnings')
plt.xlabel('budget')
plt.ylabel('Gross')
plt.show()


# In[83]:


df_sort.head()


# In[87]:


sns.regplot(x='budget',y='gross',data=df,scatter_kws={"color":"red"},line_kws={'color':"blue"})


# In[89]:


df.corr(method='pearson')


# In[92]:


correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation matrix for Numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')


# In[99]:


#Correlation of gross revenue with company
df_numerized=df

for col in df_numerized.columns:
    if(df_numerized[col].dtype=='object'):
         df_numerized[col]=df_numerized[col].astype('category')
         df_numerized[col]=df_numerized[col].cat.codes
df_numerized.head()


        


# In[103]:





# In[104]:


correlation_matrix = df_numerized.corr(method='pearson')
sns.heatmap(correlation_matrix,annot=True)
plt.title('Correlation matrix for Numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')


# In[105]:


df_numerized.corr()


# In[106]:


#Unstacking
corr_mat=df_numerized.corr()
corr_pairs=corr_mat.unstack()
corr_pairs


# In[108]:


sorted_pairs=corr_pairs.sort_values()
sorted_pairs


# In[117]:


high_corr=sorted_pairs[(sorted_pairs)>0.5]
high_corr


# In[ ]:


#Votes and budget has highest correlation and company has the lowest correlation on revenues

