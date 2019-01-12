
# coding: utf-8

# # Introduction to Pandas

# In[13]:


import pandas as pd


# ### Reading File from disk

# In[4]:


df = pd.read_csv('parks.csv', index_col=['Park Code'])


# In[6]:


df.head(3)


# ## Indexing

# ### Index single row with iloc (using row number)

# In[16]:


df.iloc[1] # at position 1


# ### Index single row with loc (using data frame index)

# In[15]:


df.loc['ACAD'] # with index = 'ACAD'


# ### Index multiple rows with iloc (using row number)

# In[21]:


df.iloc[[2, 1, 0]]


# ### Index multiple rows with loc (using data frame index)

# In[22]:


df.loc[['BADL', 'ARCH', 'ACAD']]


# In[28]:


df[:2] # indexing first 2 rows
df[2:] # indexing all rows starting from index 1
df[3:6] # indexing from row 3 to 5


# ### Indexing coloumns

# In[33]:


df['State'].head(5)
df.State.head(5)


# In[34]:


df.columns


# In[36]:


df.columns = [ col.replace(' ','_').lower() for col in df.columns]


# In[37]:


df.columns


# ### Indexing Columns and Rows

# In[39]:


df[['state','acres']][:3]


# ### Seleceting subset of data

# In[42]:


(df.state == 'UT').head(3)


# In[45]:


df[df.state == 'UT']


# #### ~ replaces not
# 
# #### | replaces or
# 
# #### & replaces and

# In[46]:


df[(df.latitude > 60) | (df.acres > 10**6)].head(3)


# ### Key Companion Methods: isin

# In[48]:


df[df.state.isin(['WA', 'OR', 'CA'])].head()


# ### Describe command

# In[49]:


df.describe()

