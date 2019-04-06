#!/usr/bin/env python
# coding: utf-8

# # Android Data from PlayStore

# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


# load data
get_ipython().run_line_magic('run', './Preprocessing.ipynb')
df = get_data()


# # Standardization of the dataset

# In[24]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scale = False
if scale :
    scaler.fit(df)
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns, index =df.index)
    


# In[25]:


df.head()


# # Exploratory plots
# We plot some data, to see its ranges

# In[49]:


features = df.columns.values
features


# In[53]:


def plot_distributions(df, features, kde = True):
    ncols = 3
    nrows = len(features) // ncols + 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols);
    # fig.suptitle('Distributions for features', fontsize = 15);
    print('Histogram of features')
    cidx = 0
    ridx = 0
    for idx, feature in enumerate(features):
        sns.distplot(df[feature], kde = kde, ax = axs[ridx][cidx] )
    #     axs[ridx][cidx].hist(df[feature])
    #     axs[ridx][cidx].set_xlabel(feature)
        if cidx == ncols - 1:
            ridx += 1
            cidx = 0
        else :
            cidx += 1
    fig.subplots_adjust(right = 2, top = 4);


# In[29]:


plot_distributions(df, features)


# In[56]:


log_features = ['reviews', 'installs', 'name_wc', 'size', 'rating']
log_df = df[log_features].apply(np.log, axis = 1)
plot_distributions(log_df, log_features)


# # To do 
# - plot some more feature pairs, try more things

# In[619]:


features


# In[620]:


plot_features = ['category', 'rating', 'reviews', 'size', 'installs', 'type', 'price_rounded','content_rating', 'genres', 'version', 'android_version',
                'name_wc']


# In[57]:


sns.pairplot(np.log(df[log_features]))


# In[594]:


sns.pairplot(np.log(df[plot_features]))


# In[595]:


plt.figure(figsize=(12,7))
ax = sns.countplot(x='category', data=df)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
_ = plt.title('App count for each category',size = 20)


# In[596]:


plt.figure(figsize=(10,8))
sns.scatterplot(x='reviews', y='rating', data=df, hue='type')


# ### Let's visualize the correlation between features
# 
# #### Correlation between "rating" and the other features

# In[46]:


correlation = df.corr()['rating']
correlation


# In[626]:


plt.figure(figsize=(15,5))
plt.ylabel('Correlation to "rating"')
plt.bar(correlation.index.values, correlation)


# **Observations**
# 
# - *rating* is correlated the most with *last_year_updated*
# - *rating* also has some intersing negative correlations
# 

# #### Correlation between all features
# - todo : enlarge plot to visualize better

# In[598]:


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

# In[599]:


# Below you can find an example that find a linear model between 'Reviews' and 'Installs' that works well
# It predicts number of 'Reviews' based on 'Installs' with an R squared error of 0.92 


# # A linear model

# In[6]:


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


# In[7]:


# when creating a ML model, we split data in train and test 
# we train the model on the train data and evaluate its performance on the test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[8]:


from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
print('Train R squared : %.4f' % lr.score(x_train,y_train))
print('Test R squared : %.4f' % lr.score(x_test,y_test))


# In[9]:


X_log = np.log(X)
Y_log = np.log(Y)
x_train, x_test, y_train, y_test = train_test_split(X_log, Y_log, test_size = 0.2, random_state = 42)


# In[10]:


lr.fit(x_train, y_train)
print('Train R squared : %.4f' % lr.score(x_train,y_train))
print('Test R squared : %.4f' % lr.score(x_test,y_test))


# In[18]:


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
ax2.set_xlabel('installs_log');
ax2.set_ylabel('reviews_log');
y_pred = lr.predict(X_log)
ax2.plot(X_log[:,0], X_log[:,0] * lr.coef_ + lr.intercept_, c = 'red');


# In[16]:


lr.coef_
lr.intercept_

