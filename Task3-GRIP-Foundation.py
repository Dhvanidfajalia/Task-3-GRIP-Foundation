#!/usr/bin/env python
# coding: utf-8

# # GRIP: THE Sparks Foundation
# 
# ## Data Science and Business Analytics Intern
# 
# ## Author: Dhvani Fajalia
# 
# ###  Task 3: Exploratory Data Analysis-Retail
# 

# ### In this we have to try to found out the weak areas where you can work to gain more profit

# ## Step 1: Importing libraries

# In[2]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Step 2: Reading the data 

# In[3]:


df = pd.read_csv("Downloads/SampleSuperstore.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# ## Step 3: Data Cleaning 

# ### Checking Duplicate values

# In[7]:


df[df.duplicated()].shape


# In[8]:


df.drop_duplicates(inplace=True)


# In[9]:


df.shape


# ### Checking Null values

# In[ ]:


df.isnull().sum()


# In[11]:


df.drop('Postal Code', axis=1, inplace=True)


# In[12]:


df.shape


# In[13]:


df.head()


# ## Step 4: EDA

# In[14]:


df['Country'].unique()


# ### Analysis of data for state 

# In[15]:


print('Number of unique states', df['State'].nunique())
df['State'].unique()


# In[16]:


plt.figure(figsize=(12,10))
sns.countplot(data=df, y='State')
plt.title('Count of the states');


# In[17]:


df['State'].value_counts().nlargest()


# In[18]:


df.groupby('State').sum()['Sales'].reset_index().sort_values(by='Sales', ascending=False)


# In[19]:


sns.heatmap(df.corr(), annot=True);


# In[20]:


plt.figure(figsize=(12,5), dpi=90)
plt.subplot(1,2,1)
sns.lineplot(data=df, x='Discount', y='Sales')
plt.title('Discount vs Sales')
plt.subplot(1,2,2)
sns.lineplot(data=df, x='Discount', y='Profit')
plt.title('Discount vs Profit')
plt.axhline(y=0,color='cyan');


# In[21]:


segments = df['Segment'].value_counts()
segments


# In[22]:


plt.figure(figsize=(12, 5), dpi=90)

plt.subplot(1, 2, 1)
sns.countplot(data=df, x='Segment')
plt.title('Segment of products')

plt.subplot(1, 2, 2)
labels = segments.index
sizes = segments.values
colors = ['lightblue', 'darkorange', 'green']
explode = (0.1, 0, 0)
plt.pie(sizes, explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
plt.axis('equal');


# ### Lets start for different sectors

# ## 1. Consumer Segment Products

# In[23]:


consumer_prod = df[df['Segment'] == 'Consumer']
consumer_prod.head()


# In[24]:


consumer_prod['State'].nunique()


# In[25]:


consumer_prod['State'].unique()


# In[26]:


all_states = df['State']


# In[27]:


print("States not buying the consumer products:")
res = all_states[~all_states.isin(consumer_prod['State'])]
print(res.unique())


# In[28]:


consumer_category = consumer_prod['Category'].value_counts()
consumer_category


# In[29]:


plt.figure(figsize=(12, 5), dpi=90)

plt.subplot(1, 2, 1)
sns.countplot(data=consumer_prod, x='Category')
plt.title('Category of Consumer segment products')

plt.subplot(1, 2, 2)
labels = consumer_category.index
sizes = consumer_category.values
colors = ['darkorange', 'green', 'lightblue']
explode = (0.1, 0, 0)
plt.pie(sizes, explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
plt.axis('equal');


# In[30]:


sns.heatmap(consumer_prod.corr(), annot=True)


# ### Visualize the same 

# In[31]:


plt.figure(figsize=(12, 5), dpi=90)

plt.subplot(1, 2, 1)
sns.lineplot(data=consumer_prod, x='Discount', y='Sales')
plt.title('Discount vs sales in Consumer segment products')

plt.subplot(1, 2, 2)
sns.lineplot(data=consumer_prod, x='Discount', y='Profit')
plt.title('Discount vs Profit in Consumer segment products')
plt.axhline(y=0, color='cyan');


# In[32]:


consumer_sub_cat = consumer_prod['Sub-Category'].value_counts()
consumer_sub_cat.shape


# In[33]:


plt.figure(figsize=(12, 7), dpi=90)

sns.countplot(data=consumer_prod, y='Sub-Category')
plt.title('Sub-Category of consumer segment products')
plt.xticks(rotation=90);


# In[34]:


plt.figure(figsize=(10, 7), dpi=70)
labels = consumer_sub_cat.index
sizes = consumer_sub_cat.values
explode = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02)
plt.pie(sizes, labels=labels, explode=explode, startangle=0, autopct='%1.1f%%', radius=1.4)
my_circle=plt.Circle((0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show();


# In[35]:


consumer_prod['State'].value_counts()


# In[36]:


consumer_region = consumer_prod['Region'].value_counts()
consumer_region


# In[37]:


plt.figure(figsize=(12,10), dpi=90)

plt.subplot(1, 2, 1)
sns.countplot(data=consumer_prod, y='State')
plt.title('State in USA buying consumer segment products')

plt.subplot(1, 2, 2)
labels = consumer_region.index
sizes= consumer_region.values
explode = (0.1, 0, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal');


# In[38]:


plt.figure(dpi=90)
sns.countplot(data=consumer_prod, x='Ship Mode')
plt.title('Shipping mode preferences of Consumer Segment');


# ## 2. Corporate segment products

# In[40]:


corporate_prod = df[df['Segment']=='Corporate']
corporate_prod.head()


# In[41]:


corporate_prod['State'].nunique()


# In[42]:


print("States not buying the Corporate products:-")
res = all_states[~all_states.isin(corporate_prod['State'])]
print(res.unique())


# In[43]:


corporate_prod['State'].value_counts()


# In[44]:


corporate_region = corporate_prod['Region']
corporate_region = corporate_region.value_counts()


# In[45]:


plt.figure(figsize=(12,10), dpi=90)

plt.subplot(1, 2, 1)
sns.countplot(data=corporate_prod, y='State')
plt.title('State in USA buying corporate segment products')

plt.subplot(1, 2, 2)
labels = corporate_region.index
sizes= corporate_region.values
explode = (0.1, 0, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal');


# In[46]:


sns.heatmap(corporate_prod.corr(), annot=True)


# In[47]:


plt.figure(figsize=(12, 5), dpi=90)

plt.subplot(1, 2, 1)
sns.lineplot(data=corporate_prod, x='Discount', y='Sales')
plt.title('Discount vs sales in Corporate segment products')

plt.subplot(1, 2, 2)
sns.lineplot(data=corporate_prod, x='Discount', y='Profit')
plt.title('Discount vs Profit in Corporate segment products')
plt.axhline(y=0, color='cyan');


# In[48]:


corporate_cat = corporate_prod['Category'].value_counts()
corporate_cat


# In[50]:


plt.figure(figsize=(12, 5), dpi=90)

plt.subplot(1, 2, 1)
sns.countplot(data=corporate_prod, x='Category')
plt.title('Category of Corporate segment products')

plt.subplot(1, 2, 2)
labels = corporate_cat.index
sizes = corporate_cat.values
colors = ['blue', 'orange', 'green']
explode = (0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal');


# ### Lets see the different sub category of different category product bought under corporate segment 

# In[52]:


corp_sub_cat_office_sup = corporate_prod[corporate_prod['Category'] == 'Office Supplies']
corp_office_sup = corp_sub_cat_office_sup['Sub-Category'].value_counts()
corp_office_sup = corp_office_sup.reset_index()
corp_office_sup


# In[53]:


corp_sub_cat_furni = corporate_prod[corporate_prod['Category'] == 'Furniture']
corp_furni = corp_sub_cat_furni['Sub-Category'].value_counts()
corp_furni = corp_furni.reset_index()
corp_furni


# In[54]:


corp_sub_cat_tech = corporate_prod[corporate_prod['Category'] == 'Technology']
corp_tech = corp_sub_cat_tech['Sub-Category'].value_counts()
corp_tech = corp_tech.reset_index()
corp_tech


# ### Vizualization of the same 

# In[58]:


plt.figure(figsize=(10.,8), dpi=90)

plt.subplot(2, 2, 1)
sns.barplot(data=corp_office_sup, x='index', y='Sub-Category')
plt.xticks(rotation=45)
plt.title('Office Supplies in Corporate Segment');

plt.subplot(2, 2, 2)
sns.barplot(data=corp_furni, x='index', y='Sub-Category')
plt.xticks(rotation=45)
plt.title('Furniture in Corporate Segment');

plt.subplot(2, 2, 3)
sns.barplot(data=corp_tech, x='index', y='Sub-Category')
plt.xticks(rotation=45)
plt.title('Technology in Corporate Segment');

plt.tight_layout()


# In[59]:


sns.countplot(data=corporate_prod, x='Ship Mode')
plt.title('Shipping mode preferences of corporate segment');


# ## 3. Home Office Segment 

# In[60]:


home_office = df[df['Segment'] == 'Home Office']
home_office.head()


# In[61]:


home_office['State'].nunique()


# In[62]:


print("States not buying the Corporate products:-")
res = all_states[~all_states.isin(home_office['State'])]
print(res.unique())


# In[63]:


homeoffice_region = home_office['Region'].value_counts()
homeoffice_region


# In[66]:


plt.figure(figsize=(12,10), dpi=90)

plt.subplot(1, 2, 1)
sns.countplot(data=home_office, y='State')
plt.title('State in USA buying Home office segment products')

plt.subplot(1, 2, 2)
labels = homeoffice_region.index
sizes = homeoffice_region.values
explode = (0.1, 0, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal');


# In[67]:


sns.heatmap(home_office.corr(), annot=True)


# ### Vizualization of the same 

# In[68]:


plt.figure(figsize=(12, 5), dpi=90)

plt.subplot(1, 2, 1)
sns.lineplot(data=home_office, x='Discount', y='Sales')
plt.title('Discount vs sales in Home Office segment')

plt.subplot(1, 2, 2)
sns.lineplot(data=home_office, x='Discount', y='Profit')
plt.title('Discount vs Profit in Home Office segment')
plt.axhline(y=0, color='cyan');


# In[69]:


homeoffice_cat = home_office['Category'].value_counts()
homeoffice_cat


# In[70]:


plt.figure(figsize=(12, 5), dpi=90)

plt.subplot(1, 2, 1)
sns.countplot(data=home_office, x='Category')
plt.title('Category of Home Office segment products')

plt.subplot(1, 2, 2)
labels = homeoffice_cat.index
sizes = homeoffice_cat.values
colors = ['blue', 'orange', 'green']
explode = (0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal');


# ### Lets see the different sub category of different products bought under Home Office Segment 

# In[81]:


homeoffice_cat_sub_cat = home_office[home_office['Category'] == 'Office Supplies']
homeoffice_sub = homeoffice_cat_sub_cat['Sub-Category'].value_counts().reset_index()

homeoffice_cat_sub_cat_furni = home_office[home_office['Category'] == 'Furniture']
homeoffice_sub_furni_furni = homeoffice_cat_sub_cat_furni['Sub-Category'].value_counts().reset_index()

homeoffice_cat_sub_cat_tech = home_office[home_office['Category'] == 'Technology']
homeoffice_sub_tech = homeoffice_cat_sub_cat_tech['Sub-Category'].value_counts().reset_index()


# In[82]:


plt.figure(figsize=(10.,8), dpi=90)

plt.subplot(2, 2, 1)
sns.barplot(data=homeoffice_sub, x='index', y='Sub-Category')
plt.xticks(rotation=45)
plt.title('Office Supplies in Home Office Segment');

plt.subplot(2, 2, 2)
sns.barplot(data=homeoffice_sub_furni_furni, x='index', y='Sub-Category')
plt.xticks(rotation=45)
plt.title('Furniture in Home Office Segment');

plt.subplot(2, 2, 3)
sns.barplot(data=homeoffice_sub_tech, x='index', y='Sub-Category')
plt.xticks(rotation=45)
plt.title('Technology in Home Office Segment');

plt.tight_layout()


# In[83]:


sns.countplot(data=home_office, x='Ship Mode')
plt.title('Shipping mode preferences of home office segment');


# ## Thank you Sparks foundation for giving this task!

# ### It has helped me to understand concept of Data Analysis in a better way.

# In[ ]:




