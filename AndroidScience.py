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


df.info()


# ## Preprocessing
# Many columns need preformatting to be able to use them in any machine learning models. They should be converted to numbers.

# In[7]:


# there are 1181 duplications in the 'name' column. What should we do about this?
df['name'].duplicated().sum()
# df[df['name'].duplicated()].sort_values('name').head(15)


# In[8]:


# preformat installs
df = df[df['installs'] != 'Free']
new_df = df['installs'].map(lambda s : s[:-1].replace(',',''))
new_df[new_df == ''] = 0
new_df.astype(int).unique()
df['installs'] = new_df.astype(int)


# In[9]:


# preformat reviews
df['reviews'] = df['reviews'].astype(int)


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


# preformat price
# df['price'].unique()
temp_df = df['price'].apply(lambda s: s.replace('$', ''))
temp_df = temp_df.astype(float)
df['price'] = temp_df


# In[12]:


# preformat type
df['type'].unique()

# there is one app that doesn't have type -> drop it
# df['type'].isnull().sum()
df.dropna(subset=['type'], inplace=True)
# df['type'].isnull().sum()

# convert 'type' column to category
df['type'] = pd.Categorical(df['type'])


# In[13]:


# preformat category
df['category'] = pd.Categorical(df['category'])


# In[14]:


# preformat content_rating
df['content_rating'] = pd.Categorical(df['content_rating'])


# In[15]:


# preformat genres
# there are 119 unique genres
df['genres'].nunique()

# keep only the first genre
df['genres'].value_counts().tail(10)

# there are 498 apps that have two genres
df['genres'].str.contains(';').sum()

df['genres'] = df['genres'].str.split(';').str[0]
df['genres'] = pd.Categorical(df['genres'])

# we are down to 48 unique genres
df['genres'].nunique()


# In[16]:


# preformat last_updated -> convert it to difference in days
df['last_updated'] = pd.to_datetime(df['last_updated'])

# consider the max last_updated in df, the date of reference for the other last_updated 
# last_updated will become a negative integer (the number of days between that date and date of reference)
df['last_updated'] = (df['last_updated'] - df['last_updated'].max()).dt.days


# ## Feature engineering
# Features below are derived from the original features of data

# In[17]:


# preprocess name
# keep the word count of the app name
df['name_wc'] = df['name'].apply(lambda s : len(s.replace('&','').replace('-', '').split()))


# In[18]:


df.head()


# In[19]:


# preprocess version & android_version
def vs_transform(version):
    if version == 'Varies with device':
        return -1
    if version == np.NaN or version == np.nan:
        return np.nan
    return version[0]
# there are some edge cases that still need to be cared about
# df['version'].astype(str).sort_values()[-1600:]

# df['major_version'] = df['version'].astype(str).apply(vs_transform).astype(int)
# df['android_version'].astype(str).apply(vs_transform).astype(int)


# In[20]:


# drop columns not used
drop_columns = ['name', 'version', 'android_version']
df.drop(columns = drop_columns, inplace = True)


# In[21]:


df.head()


# ## Missing values
# 
# Rating column has 10% missing values. To not lose the data, we try and predict its values using the other features.

# In[22]:


# check for null values
# rating has a few
df.isnull().sum()


# In[23]:


# get the rows with null ratings out, to predict them later
to_predict_rating = df[df['rating'].isnull()]

df = df.dropna()


# # Exploratory plots
# We plot some data, to see its ranges

# In[25]:


fig, axs = plt.subplots(nrows = 2, ncols = 3);
fig.suptitle('Histogram of values for some features', fontsize = 15);
axs[0][0].hist(df['rating']);
axs[0][0].set_xlabel('rating');
axs[0][1].hist(df['reviews']);
axs[0][1].set_xlabel('reviews');
axs[0][2].hist(df['installs']);
axs[0][2].set_xlabel('installs');
axs[1][0].hist(df['size']);
axs[1][0].set_xlabel('size');
axs[1][1].hist(df['name_wc']);
axs[1][1].set_xlabel('App Name WordCount');
fig.subplots_adjust(right = 2);


# In[25]:


sns.pairplot(df)


# In[26]:


plt.figure(figsize=(12,7))
ax = sns.countplot(x='category', data=df)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
_ = plt.title('App count for each category',size = 20)


# In[27]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='rating', y='category', data=df, hue='type')


# ### Let's visualize the correlation between features
# 
# #### Correlation between "rating" and the other features

# In[28]:


correlation = df.corr()['rating']
plt.figure(figsize=(15,5))
plt.ylabel('Correlation to "rating"')

plt.bar(correlation.index.values, correlation)


# **Observations**
# 
# - *rating* is correlated the most with *last_year_updated*
# 

# #### Correlation between all features

# In[30]:


k = len(df.columns.values) #number of variables for heatmap
cols = df.corr().nlargest(k, 'rating')['rating'].index
cm = df[cols].corr()

# enlarge plot
plt.figure(figsize=(10, 8), dpi= 80, facecolor='w', edgecolor='k')

# plot heatmap
sns.heatmap(cm, annot=True, cmap = 'coolwarm')


# **Observations**
# 
# - *reviews* and *installs* are strongly correlated
# - *size* and *last_year_updated* may be somehow related

# In[27]:


# Because not all features are preprocessed yet, we got only to use 'Reviews' and 'Installs'. 
# This led to terrible results, like an R squared error of 0.002
# But below you can find an example that find a linear model between 'Reviews' and 'Installs' that works well
# It predicts number of 'Reviews' based on 'Installs' with an R squared error of 0.92 


# # A linear model

# In[31]:


# convert categorical columns to int so that they can be used by ML models
cat_columns = df.select_dtypes(['category']).columns
cat_columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)


# In[32]:


# we use .values because the ML models work with numpy arrays, not pandas dataframes
Y = df['reviews'].values
X = df[['installs']].values


# In[33]:


# In some cases we may need to scale data. There are many types of scallers in the preprocessing module. 
# Here is an example

# from sklearn import preprocessing
# scaler = preprocessing.MinMaxScaler()
# X = scaler.fit_transform(X)
# Y = scaler.fit_transform(Y.reshape(-1,1)).squeeze()


# In[34]:


# when creating a ML model, we split data in train and test 
# we train the model on the train data and evaluate its performance on the test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[35]:


from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
print('Train R squared : %.4f' % lr.score(x_train,y_train))
print('Test R squared : %.4f' % lr.score(x_test,y_test))


# In[36]:


X_log = np.log(X)
Y_log = np.log(Y)
x_train, x_test, y_train, y_test = train_test_split(X_log, Y_log, test_size = 0.2, random_state = 42)


# In[37]:


lr.fit(x_train, y_train)
print('Train R squared : %.4f' % lr.score(x_train,y_train))
print('Test R squared : %.4f' % lr.score(x_test,y_test))


# In[38]:


df.columns
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2);
fig.suptitle('Linear model between reviews and installs', fontsize = 15)
fig.subplots_adjust(right = 2)
ax1.set_title('Original data')
ax1.scatter(X, Y);
ax1.set_xlabel('installs');
ax1.set_ylabel('reviews');
ax2.set_title('Log data')
ax2.scatter(X_log[:,0], Y_log);
ax2.set_xlabel('installs_log')
ax2.set_ylabel('reviews_log');
y_pred = lr.predict(X_log)
ax2.plot(X_log[:,0], y_pred, c = 'red');


# ## Trying other modules
# 
# We transformed the problem in a classification one. Now rating can be *poor* (< 4) and *excellent* (>=4). We have around 57 % accuracy on the test data. 
# 
# **Next thing** : We should try adding or changing the features of data, and try more values for the hyperparameters of the algorithm

# In[141]:


Y = df['rating'].values
X = df[['size', 'name_wc', 'price', 'type']]


# In[142]:


Y = pd.cut(Y, 
           bins=[0, 4, 5], 
           labels=[0, 1])
Y.value_counts()


# In[143]:


# the dataset is rather imbalanced, which will skew the results. So we reduce the number of big rating examples
# we can also try upsampling the small rating examples
from imblearn.under_sampling import RandomUnderSampler
usampler = RandomUnderSampler(random_state = 42)
X, Y = usampler.fit_resample(X,Y)


# In[144]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[145]:


from sklearn import ensemble, tree, svm, neighbors


from model import Model, ModelsBenchmark

models = [svm.SVC(),
          tree.DecisionTreeClassifier(),
          ensemble.RandomForestClassifier(),
          neighbors.KNeighborsClassifier()
         ]

bench = ModelsBenchmark(models)
benchmarks = bench.compute_scores((x_train, y_train), (x_test, y_test))
benchmarks


# In[146]:


# print and plot metrics for the best one
from sklearn.metrics import confusion_matrix
test_score = benchmarks[0][0]
clf = benchmarks[0][1]
y_pred = clf.predict(x_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cnf_matrix)
plt.xlabel('Predicted')
plt.ylabel('Actual')
print(cnf_matrix)
print('Accuracy : %.2f ' % test_score)

