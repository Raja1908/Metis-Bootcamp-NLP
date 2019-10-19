#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('test.csv')
df.head()


# In[2]:


df.shape


# In[3]:


print(df.shape[0])
print(df.shape[1])
print(df.columns.tolist())
print(df.isnull().sum().values.sum())
print(df.nunique())


# In[5]:


df.info()
df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df = df[['brand', 'categories', 'name', 'reviews.date', 'reviews.dateAdded', 'reviews.dateSeen', 
          'reviews.didPurchase', 'reviews.doRecommend', 'reviews.id', 'reviews.numHelpful', 'reviews.rating', 
          'reviews.sourceURLs', 'reviews.text', 'reviews.title', 'reviews.username']]

# rename relevant columns
df.columns = ['brand', 'categories', 'product', 'reviews_date', 'reviews_dateAdded', 'reviews_dateSeen', 
          'reviews_didPurchase', 'reviews_doRecommend', 'reviews_id', 'reviews_numHelpful', 'reviews_rating', 
          'reviews_sourceURLs', 'reviews_text', 'reviews_title', 'reviews_username'] 


# In[8]:


df


# In[10]:


df = df[df['reviews_didPurchase'] == True]


# In[12]:


df[df['reviews_text'].str.contains('disappoint')]


# In[13]:


df['product'].value_counts()


# In[15]:


df['product'].value_counts()[0:10].plot('barh', figsize=[10,6], fontsize = 20).invert_yaxis()


# In[16]:


df[df['product'] == 'The Foodsaver174 10 Cup Fresh Container - Fac10-000']


# In[17]:


df_user_product = df[['reviews_username', 'product', 'reviews_rating']]
df_user_product = df_user_product[df_user_product['reviews_rating']==5]
df_user_product = df_user_product.drop(columns = ['reviews_rating'], axis=1)
df_user_product


# In[18]:


df_user_product.to_csv('df_user_product.csv', index=False)


# In[19]:


df = pd.read_csv('df_user_product.csv')


# In[20]:


from collections import defaultdict
from collections import Counter
import csv
pd.read_csv('df_user_product_dummy_users.csv').head(10)


# In[21]:


user_product_map = defaultdict(list)
product_user_map = defaultdict(list)


# In[27]:


with open('df_user_product_dummy_users.csv', 'r') as csvfile:
    w = csv.reader(csvfile, delimiter = ',')
    for row in w:
        user_product_map[row[0]].append(row[1])
        product_user_map[row[1]].append(row[0])


# In[28]:


df_user_product['reviews_username'].value_counts()


# In[29]:


user_product_map['user05']


# In[30]:


user_product_map['Donna']


# In[33]:


product_user_map['Pantene Pro-V Expert Collection Age Defy Conditioner']
product_user_map["Chuggington: Let's Ride The Rails"]


# In[43]:


def get_similar_product(user_product_map, product_user_map, m):
    biglist = []
    for u in product_user_map[m]:
        biglist.extend(user_product_map[u])
    return Counter(biglist).most_common(11)[1:]


# In[49]:


def get_product_recommendation(user_product_map, product_user_map, u1):
    biglist = []
    for m in user_product_map[u1]:
        for u in product_user_map[m]:
            biglist.extend(user_product_map[u])
    return Counter(biglist).most_common(10)


# In[45]:


get_similar_product(user_product_map, product_user_map, 'product01')


# In[46]:


get_similar_product(user_product_map, product_user_map, "Pantene Pro-V Expert Collection Age Defy Conditioner")  # product 


# In[47]:


get_similar_product(user_product_map, product_user_map, 'K-Y Love Sensuality Pleasure Gel') 


# In[50]:


get_product_recommendation(user_product_map, product_user_map, 'An anonymous customer')  # user 'Joshua'


# In[51]:


get_product_recommendation(user_product_map, product_user_map, 'Cassie')


# In[52]:


get_product_recommendation(user_product_map, product_user_map, 'Michael')


# In[53]:


get_product_recommendation(user_product_map, product_user_map, 'user03')


# In[ ]:




