#!/usr/bin/env python
# coding: utf-8

# # Android Data from PlayStore

# In[1]:


get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

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
    'Reviews': 'reviews',
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
    columns={'App': 'app_name', 'Translated_Review': 'review', 'Sentiment': 'sentiment',
             'Sentiment_Polarity': 'polarity', 'Sentiment_Subjectivity': 'subjectivity'}
    , inplace=True)
df_user_reviews.head()

# In[5]:


df.info()

# In[6]:


df.describe()

# ## Preprocessing
# Many columns need preformatting to be able to use them in any machine learning models. They should be converted to numbers.

# In[7]:


# preformat installs
df = df[df['installs'] != 'Free']
new_df = df['installs'].map(lambda s: s[:-1].replace(',', ''))
new_df[new_df == ''] = 0
new_df.astype(int).unique()
df['installs'] = new_df.astype(int)

# In[8]:


# preformat reviews
df['reviews'] = df['reviews'].astype(int)


# In[9]:


# Other preformat cells here !!!!!!!


# ## Feature engineering
# Features below are derived from the original features of data

# In[10]:


# preformat size
# np.sort(df['size'].unique())
def size_transform(size):
    if size == 'Varies with device':
        return 1
    unit = size[-1]
    number = float(size[:-1])
    if unit == 'M':
        return number * 1024 * 1024
    if unit == 'k':
        return number * 1024


df['size'] = df['size'].apply(size_transform).astype(int)

# In[11]:


# preprocess last_updated
# keep only the year
df['last_year_updated'] = df['last_updated'].apply(lambda s: s[-4:]).astype(int)

# todo: maybe convert this column to datetime object


# In[12]:


# preprocess name
# keep the word count of the app name
df['name_wc'] = df['name'].apply(lambda s: len(s.replace('&', '').replace('-', '').split()))


# In[13]:


# preprocess version & android_version
def vs_transform(version):
    if version == 'Varies with device':
        return -1
    if version == np.NaN or version == np.nan:
        return np.nan
    return version[0]


# there are some edge cases that still need to be cared about
df['version'].astype(str).sort_values()[-1600:]
# df['major_version'] = df['version'].astype(str).apply(vs_transform).astype(int)
# df['android_version'].astype(str).apply(vs_transform).astype(int)


# In[14]:


# drop columns not used
orig_df = df.copy()
drop_columns = ['name', 'last_updated', 'version', 'android_version']
df.drop(columns=drop_columns, inplace=True)

# In[15]:


df.head()

# ## Missing values
#
# Rating column has 10% missing values. To not lose the data, we try and predict its values using the other features.

# In[16]:


# check for null values
# rating has a few
df.isnull().sum()

# In[17]:


# get the rows with null ratings out, to predict them later
to_predict_rating = df[df['rating'].isnull()]
# dfn - df without nulls
dfn = df[~df['rating'].isnull()]
dfn.shape
dfn.info()
dfn.describe()

# # Exploratory plots
# We plot some data, to see its ranges

# In[18]:


fig, axs = plt.subplots(nrows=2, ncols=3);
fig.suptitle('Histogram of values for some features', fontsize=15);
axs[0][0].hist(dfn['rating']);
axs[0][0].set_xlabel('rating');
axs[0][1].hist(dfn['reviews']);
axs[0][1].set_xlabel('reviews');
axs[0][2].hist(dfn['installs']);
axs[0][2].set_xlabel('installs');
axs[1][0].hist(dfn['size']);
axs[1][0].set_xlabel('size');
axs[1][1].hist(dfn['name_wc']);
axs[1][1].set_xlabel('App Name WordCount');
fig.subplots_adjust(right=2);

# ### Let's visualize the correlation between features
#
# #### Correlation between "rating" and the other features

# In[19]:


correlation = dfn.corr()['rating']
plt.figure(figsize=(15, 5))
plt.ylabel('Correlation to "rating"')

plt.bar(correlation.index.values, correlation)

# **Observations**
#
# - *rating* is correlated the most with *last_year_updated*
#

# #### Correlation between all features

# In[20]:


k = len(dfn.columns.values)  # number of variables for heatmap
cols = dfn.corr().nlargest(k, 'rating')['rating'].index
cm = dfn[cols].corr()

# enlarge plot
plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

# plot heatmap
sns.heatmap(cm, annot=True, cmap='viridis')

# **Observations**
#
# - *reviews* and *installs* are strongly correlated
# - *size* and *last_year_updated* may be somehow related

# In[21]:


# Because not all features are preprocessed yet, we got only to use 'Reviews' and 'Installs'.
# This led to terrible results, like an R squared error of 0.002
# But below you can find an example that find a linear model between 'Reviews' and 'Installs' that works well
# It predicts number of 'Reviews' based on 'Installs' with an R squared error of 0.92


# # A linear model

# In[22]:


# we use .values because the ML models work with numpy arrays, not pandas dataframes
Y = dfn['reviews'].values
X = dfn[['installs']].values

# In[23]:


# In some cases we may need to scale data. There are many types of scallers in the preprocessing module.
# Here is an example

# from sklearn import preprocessing
# scaler = preprocessing.MinMaxScaler()
# X = scaler.fit_transform(X)
# Y = scaler.fit_transform(Y.reshape(-1,1)).squeeze()


# In[ ]:


# when creating a ML model, we split data in train and test
# we train the model on the train data and evaluate its performance on the test data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# In[ ]:


from sklearn import linear_model

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
print('Train R squared : %.4f' % lr.score(x_train, y_train))
print('Test R squared : %.4f' % lr.score(x_test, y_test))

# In[ ]:


X_log = np.log(X)
Y_log = np.log(Y)
x_train, x_test, y_train, y_test = train_test_split(X_log, Y_log, test_size=0.2, random_state=42)

# In[ ]:


lr.fit(x_train, y_train)
print('Train R squared : %.4f' % lr.score(x_train, y_train))
print('Test R squared : %.4f' % lr.score(x_test, y_test))

# In[ ]:


dfn.columns
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2);
fig.suptitle('Linear model between reviews and installs', fontsize=15)
fig.subplots_adjust(right=2)
ax1.set_title('Original data')
ax1.scatter(X, Y);
ax1.set_xlabel('installs');
ax1.set_ylabel('reviews');
ax2.set_title('Log data')
ax2.scatter(X_log[:, 0], Y_log);
ax2.set_xlabel('installs_log')
ax2.set_ylabel('reviews_log');
y_pred = lr.predict(X_log)
ax2.plot(X_log[:, 0], y_pred, c='red');

# ## Trying other modules

# In[ ]:


Y = dfn['rating'].values
X = dfn[['size', 'installs', 'reviews', 'last_year_updated', 'name_wc']]

# In[ ]:


from sklearn import svm
from model import Model

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
svr = Model(svm.SVR())
train_score, test_score = svr.use_model((x_train, y_train), (x_test, y_test))
print('Train score : %.4f' % train_score)
print('Test score : %.4f' % test_score)
