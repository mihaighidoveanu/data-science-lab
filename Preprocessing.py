#!/usr/bin/env python
# coding: utf-8

# # Android Data from PlayStore

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Load data
df = pd.read_csv('googleplaystore.csv')
df_user_reviews = pd.read_csv('googleplaystore_user_reviews.csv')
df.columns
df_user_reviews.columns


# In[4]:


# rename columns
df.rename(columns={
    'App': 'name',
    'Category': 'category',
    'Rating': 'rating',
    'Reviews' :'reviews',
    'Size': 'size',
    'Installs': 'installs',
    'Type': 'type',
    'Price': 'price',
    'Content Rating': 'content_rating',
    'Genres': 'genres',
    'Last Updated': 'last_updated',
    'Current Ver': 'version',
    'Android Ver': 'android_version'
    
}, inplace=True)
df.head()

df_user_reviews.rename(
    columns={'App':'app_name', 'Translated_Review': 'review', 'Sentiment': 'sentiment', 'Sentiment_Polarity': 'polarity', 'Sentiment_Subjectivity': 'subjectivity'}
, inplace=True)
df_user_reviews.head()


# In[5]:


orig_df = df.copy()


# In[6]:


df = orig_df.copy()


# In[7]:


df.info()


# ## Preprocessing
# Many columns need preformatting to be able to use them in any machine learning models. They should be converted to numbers.

# In[8]:


# there are 1181 duplications in the 'name' column. What should we do about this?
print('Number of duplicate entries : %d' % df.duplicated(subset = ['name']).sum())
# remove duplicates 
df = df[~df.duplicated(subset = ['name'], keep = 'first')]


# In[9]:


# preformat installs
df = df[df['installs'] != 'Free']
new_df = df['installs'].map(lambda s : s[:-1].replace(',',''))
new_df[new_df == ''] = 0
df['installs'] = new_df.astype(int)
df['installs'].unique()


# In[10]:


# preformat reviews
df['reviews'] = df['reviews'].astype(int)
df['reviews'].head()


# In[11]:


# preformat size
# np.sort(df['size'].unique())
# transform sizes to kb units
def size_transform(size):
    if size == 'Varies with device':
        return 1
    unit = size[-1]
    number = float(size[:-1])
    if unit == 'M':
        return number * 1000
    if unit == 'k':
        return number
df['size'].head()
df['size'] = df['size'].apply(size_transform).astype(int)
df['size'].head()


# In[12]:


# preformat price
# df['price'].unique()
temp_df = df['price'].apply(lambda s: s.replace('$', ''))
df['price'] = temp_df.astype(float)
df['price'].unique()


# In[13]:


df['price_rounded'] = df['price'].apply(np.round)


# ### Categorical values

# In[14]:


categories = {}


def categorize(df, columns):
    if isinstance(columns, str):
        column = columns
        df[column], cats = pd.factorize(df[column])
        categories[column] = cats
    else :
        for column in columns:
            df[column], cats = pd.factorize(df[column])
            categories[column] = cats
    return df, cats


# In[15]:


# cat_columns = ['type', 'category', 'content_rating', 'genres' ]
# df, _ = categorize(df, cat_columns)


# In[16]:


# preformat type
print('Original values : %s' % df['type'].unique())

# there is one app that doesn't have type -> drop it
# df['type'].isnull().sum()
df.dropna(subset=['type'], inplace=True)
# df['type'].isnull().sum()

# convert 'type' column to category
df, _ = categorize(df,'type')
df['type'].unique()
categories['type']


# In[17]:


# preformat category
df, _ = categorize(df, 'category')
df['category'].unique()
categories['category']


# In[18]:


# preformat content_rating
df, _ = categorize(df, 'content_rating')
df['content_rating'].unique()
categories['content_rating']


# In[19]:


# preformat genres
# there are 119 unique genres
print('Unique genres before preprocessing : %d' % df['genres'].nunique())

# keep only the first genre
df['genres'].value_counts().tail(10)

# there are 498 apps that have two genres
print(' Apps with more than one genre : %d ' % df['genres'].str.contains(';').sum())

df['genres'] = df['genres'].str.split(';').str[0]

df, _ = categorize(df, 'genres')
# we are down to 48 unique genres
print('Unique genres : %d' % df['genres'].nunique())
df['genres'].unique()
categories['genres']


# ## Datetime features

# In[20]:


# preformat last_updated -> convert it to difference in days
df['last_updated'] = pd.to_datetime(df['last_updated'])

# consider the max last_updated in df, the date of reference for the other last_updated 
# last_updated will become a negative integer (the number of days between that date and date of reference)
df['last_updated_days'] = (df['last_updated'].max() - df['last_updated']).dt.days


# In[21]:


df['last_updated_year'] = df['last_updated'].dt.year


# In[22]:


print('Oldest updated app : %d' % df['last_updated_year'].min())


# In[23]:


df['last_updated_month'] = df['last_updated'].dt.month


# In[24]:


df['last_updated_day'] = df['last_updated'].dt.day


# In[25]:


df['last_updated_year'] = df['last_updated_year'].max() - df['last_updated_year']


# In[26]:


df['last_updated_month_sin'] = np.sin((df['last_updated_month']-1)*(2.*np.pi/12))
df['last_updated_month_cos'] = np.cos((df['last_updated_month']-1)*(2.*np.pi/12))


# ## Feature engineering
# Features below are derived from the original features of data

# In[27]:


# preprocess name
# keep the word count of the app name
import string

def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))

df['name_wc'] = df['name'].apply(lambda s : len(remove_punctuation(s).split()))


# In[28]:


print('Longest app name with %d words : %s' % (df['name_wc'].max(), df['name'][df['name_wc'].idxmax()]))
print('Shortes app name with %d words : %s' % (df['name_wc'].min(), df['name'][df['name_wc'].idxmin()]))


# In[29]:


def vs_extract(version_str):
    for char in version_str:
        try:
            version = float(char)
            return version
        except :
            pass
    return np.NaN
# preprocess version & android_version
def vs_transform(version):
    if version.lower() == 'varies with device':
        return np.NaN
    if version == np.NaN or version == np.nan or version == 'nan':
        return np.NaN
#     try :
    version = vs_extract(version.split('.')[0])
#     except :
#         print(version)        
    return version
# there are some edge cases that still need to be cared about
# df['version'].astype(str).sort_values()[-1600:]


# transform this to int 


# In[30]:


df['version'] = df['version'].astype(str).apply(vs_transform).astype(float)


# In[31]:


df['android_version'] = df['android_version'].astype(str).apply(vs_transform).astype(float)


# In[32]:


# drop columns not used
drop_columns = ['name', 'last_updated', 'last_updated_month']
df.drop(columns = drop_columns, inplace = True)


# ## Missing values
# 
# Rating column has 10% missing values. To not lose the data, we try and predict its values using the other features.

# In[33]:


# check for null values
# rating has a few
# versions have a few nulls also
df.isnull().sum()


# In[34]:


# get the rows with null ratings out, to predict them later
to_predict_rating = df[df['rating'].isnull()]


# In[35]:


df = df.dropna()
df.isnull().sum()


# In[36]:


len(df)


# In[37]:


# save the preprocessed version
ready_df = df.copy()


# In[38]:


df = ready_df.copy()


# In[39]:


df.shape


# In[2]:


print('Ignore all the gibberish above ! Dataframe is loaded ! ')


# In[1]:


def get_data(preprocessed = True):
    if preprocessed:
        return ready_df
    else:
        return orig_df

