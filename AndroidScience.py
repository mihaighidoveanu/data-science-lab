#!/usr/bin/env python
# coding: utf-8

# # Android Data from PlayStore

# In[213]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[214]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[215]:


# load data
get_ipython().run_line_magic('run', './Preprocessing.ipynb')
df = get_data()
orig_df = df.copy()


# # Science
# 
# We transformed the problem in a classification one. Now rating can be *poor* (< 4) and *excellent* (>=4). 
# 
# We achieved the following : 
# - 70% with RandomForest(n_estimators = 100)
# 
# Notes : 
# - RandomForest clearly tends to overfit. Reducing the complexity of the tree algorithm doesn't improve accuracy in cross validation by no means. 
#     + this may mean that the model is too complex. reducing the number of features took into account can help
# **Next thing** : We should try adding or changing the features of data, and try more values for the hyperparameters of the algorithm

# In[312]:


df = orig_df.copy()


# In[313]:


features = df.columns.values
features


# In[314]:


pre_features = ['category', 'size', 'type', 'price', 'content_rating', 'genres', 'android_version', 'name_wc']
post_features = [feature for feature in df.columns.values if feature not in pre_features]
log_features = ['reviews', 'installs', 'name_wc', 'size', 'rating']


# In[315]:


df[log_features] = np.log(df[log_features])


# In[316]:


sns.distplot(df['installs'])


# In[317]:


from scipy import stats
remove_outliers = True
if remove_outliers:
    z = np.abs(stats.zscore(df))
    mask = (z>3).all(axis = 1)
    print('Removing %d outliers' % mask.sum())
    df = df[~mask]


# In[318]:


df['installs'].describe()


# In[319]:


Y = df['installs'].values
X = df.drop(columns = ['installs'])
X = X[pre_features].values


# In[321]:


# split rating into two labels
bins = [Y.min(), np.percentile(Y, 75), Y.max()]
bins


# In[322]:


Y[Y < bins[1]] = 0
Y[Y >= bins[1]] = 1


# In[327]:


(Y == 0).sum()
(Y == 1).sum()


# In[328]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)


# In[326]:


from sklearn.linear_model import Ridge
reg = Ridge()
reg.fit(x_train, y_train)
reg.score(x_train, y_train)
sns.regplot(y_train, reg.predict(x_train))


# In[329]:


# the dataset is rather imbalanced, which will skew the results. So we reduce the number of big rating examples
# we can also try upsampling the small rating examples
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
# from imblearn.usampler import TODO
sampler = RandomOverSampler(random_state = 42)
x_train, y_train = sampler.fit_resample(x_train,y_train)
len(y_train[y_train==1])
len(y_train[y_train==0])


# In[330]:


from sklearn import preprocessing
# scale data if needed. forests and trees don't need it. Others do. 
scale = True
if scale:
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


# In[331]:


from sklearn import ensemble, tree, svm, neighbors
from sklearn.model_selection import cross_val_score
from model import Model, ModelsBenchmark

models = [
        svm.SVC(kernel = 'rbf', random_state=42),
        tree.DecisionTreeClassifier( min_impurity_decrease = 0, min_samples_leaf = 1, random_state = 42),
        ensemble.RandomForestClassifier(n_estimators=100, min_impurity_decrease=0, min_samples_leaf=1, random_state=42),
        neighbors.KNeighborsClassifier(n_neighbors = 2)
         ]
bench = ModelsBenchmark(models);
bench.fit(x_train, y_train);
bench.score(x_train, y_train);
bench.score(x_test, y_test);
bench._scores


# In[332]:


# hyperparameters will be tuned agains the following score mean. 
# This is called cross-validation and is done to avoid overfitting the test data
clf = ensemble.RandomForestClassifier(n_estimators=100, min_impurity_decrease=0, min_samples_leaf=1, random_state=42)
scores = cross_val_score(clf, x_train, y_train, cv = 3)
scores
scores.mean()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)


# # Evaluating models

# In[333]:


# reduce dimensionality to be able to plot data
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, random_state = 42);
    
def reduce_dimensions(X, fit=False):
    if fit:
        pca.fit(X)
    return pca.transform(X)


# In[334]:


# print and plot metrics for the best one
from sklearn.metrics import confusion_matrix
bench.fit(x_train, y_train)
bench.score(x_test, y_test)
clf = bench._sorted[0]
fig, axs = plt.subplots(nrows = 1, ncols = 4);
fig.subplots_adjust(right = 2);
x_plot = reduce_dimensions(x_train, fit = True)
axs[0].scatter(x_plot[:, 0], x_plot[:,1], c = y_train);
axs[0].set_title('Train Data');
axs[0].set_xlabel('x_0');
axs[0].set_ylabel('x_1');
x_plot = reduce_dimensions(x_test)
axs[1].scatter(x_plot[:, 0], x_plot[:,1], c = y_test);
axs[1].set_title('Test Data');
axs[1].set_xlabel('x_0');
axs[1].set_ylabel('x_1');
test_cnf_matrix = confusion_matrix(y_test, clf.predict(x_test))
sns.heatmap(test_cnf_matrix, ax = axs[2], vmin = 0);
axs[2].set_title('Test');
axs[2].set_xlabel('Predicted');
axs[2].set_ylabel('Actual');
train_cnf_matrix = confusion_matrix(y_train, clf.predict(x_train))
sns.heatmap(train_cnf_matrix, ax = axs[3], vmin = 0);
axs[3].set_title('Train');
axs[3].set_xlabel('Predicted');
axs[3].set_ylabel('Actual');
print(train_cnf_matrix)
print('Train Accuracy : %.2f ' % clf.score(x_train, y_train))
print('Test Accuracy : %.2f ' % clf.score(x_test, y_test))


# In[335]:


clf = ensemble.RandomForestClassifier(n_estimators=100, min_impurity_decrease=0, min_samples_leaf=1, random_state=42)


# In[337]:


# plot both labels separately and our predictions on them
fig, axs = plt.subplots(nrows = 1, ncols = 2)
x_plot_train = reduce_dimensions(x_train, fit=True)
y_plot_train = y_train
x_plot_test = reduce_dimensions(x_test)
y_plot_test = y_test
clf.fit(x_plot_train,y_plot_train)
x_plot = x_plot_test
y_plot = y_plot_test
y_pred = clf.predict(x_plot)
fig.subplots_adjust(right = 2)
labeled_0 = x_plot[y_plot == 0]
scatter = axs[0].scatter(labeled_0[:,0], labeled_0[:, 1], c = y_pred[y_plot ==  0], )
fig.colorbar(scatter, ax = axs[0])
axs[0].set_title('Points with true label 0 ')
labeled_0_acc = (y_pred == 0) == (y_plot == 0).sum() / len(y_pred)
# print('Accuracy for true label 0 : %.3f' % labeled_0_acc)
labeled_1 = x_plot[y_plot == 1]
scatter = axs[1].scatter(labeled_1[:,0], labeled_1[:, 1], c = y_pred[y_plot == 1] )
fig.colorbar(scatter, ax = axs[1])
axs[1].set_title('Points with true label 1 ')


# In[73]:


y_pred == 0


# In[57]:


# print some correctly and incorrectly labeled data
from random import randint
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
# correct = x_test[y_pred == y_test]
# incorrect = x_test[~(y_pred == y_test)]

columns = df.columns.values
def get_samples(x, y_true, y_pred, sample_type = 'correct', count = 5):
    mask = (y_pred == y_true)
    if sample_type == 'incorrect':
        mask = ~mask
    x = x[mask]
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    df = pd.DataFrame(columns=[*columns , 'predicted', 'true'])
    if len(x) == 0:
        return df
    for _ in range(count):
        idx = randint(0, len(x)) 
        dct = {}
        dct['predicted'] = y_pred[idx]
        dct['true'] = y_true[idx]
        for i in range(x.shape[1]):
            dct[df.columns.values[i]] = x[idx][i]
        df = df.append(dct,
                       ignore_index=True)
    return df

print("====Correct samples =====")
get_samples(x_test, y_test, y_pred, sample_type='correct') 
print("====Incorrect samples =====")
get_samples(x_test, y_test, y_pred, sample_type='incorrect') 


# In[62]:


# Visualisation of the decision tree created by the algorithm, for fun and insight
import graphviz
from sklearn.tree import export_graphviz
clf = Model(tree.DecisionTreeClassifier(max_depth = 5))
clf.compute_scores((x_train, y_train), (x_test, y_test))
clf.model.tree_.max_depth
dot_data = export_graphviz(clf.model,
                           out_file=None,
                           feature_names=pre_features,
                           class_names=['fair', 'excellent'],
                           filled=True,
                           rounded=True)
graph = graphviz.Source(dot_data)
graph


# # Neural network model

# In[188]:


import keras
from keras import layers

num_classes = 2
input_shape = x_train.shape[1]
model = keras.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(input_shape,)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.BatchNormalization(input_shape=(input_shape,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()


# In[189]:


model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])


# In[190]:


from keras.utils import to_categorical
no_epochs = 20
batch_size = 1024
history = model.fit(x_train, to_categorical(y_train), batch_size = batch_size, epochs = no_epochs )


# In[171]:


model.evaluate(x_test, to_categorical(y_test))


# In[186]:


df = pd.DataFrame({'epochs':history.epoch, 'loss': history.history['loss'], 
#                    'validation_loss': history.history['val_loss']
                  })
g = sns.pointplot(x="epochs", y="loss", data=df, fit_reg=False, color = 'yellow')
# g = sns.pointplot(x="epochs", y="validation_loss", data=df, fit_reg=False, color='red')


# In[172]:


import seaborn as sns
df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc']
#                    , 'validation_accuracy': history.history['val_acc']
                  })
g = sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
# g = sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')

