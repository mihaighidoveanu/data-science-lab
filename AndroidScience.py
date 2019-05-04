#!/usr/bin/env python
# coding: utf-8

# # Android Data from PlayStore

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# load data
get_ipython().run_line_magic('run', './Preprocessing.ipynb')
df = get_data();
orig_df = df.copy();


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

# In[ ]:


df = orig_df.copy()


# In[ ]:


features = df.columns.values.tolist()
print(features)


# In[ ]:


pre_features = ['category', 'size', 'type', 'price', 'content_rating', 'genres', 'android_version', 'name_wc']
post_features = [feature for feature in df.columns.values if feature not in pre_features]
log_features = ['reviews', 'installs', 'name_wc', 'size', 'rating']
cat_features = ['category', 'type', 'content_rating', 'genres']


# In[ ]:


use_categories = True
if use_categories:
    pre_features = list(set(pre_features + cat_features))
    post_features = list(set(post_features + cat_features))
else:
    pre_features = [ft for ft in pre_features if ft not in cat_features]
    post_features = [ft for ft in post_features if ft not in cat_features]
print('Predictors are : %s' % pre_features)


# In[ ]:


use_log = False
if use_log:
    df[log_features] = np.log(df[log_features])


# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, random_state = 42)
train.shape
test.shape


# ## Preprocessing
# Missing values and outliers removal

# In[ ]:


from OutlierIQR import OutlierIQR
to_remove_outliers = False
outliers_cols = ['installs']
if to_remove_outliers:
    odetector = OutlierIQR()
    odetector.fit(train, columns = ['installs'])
    train = odetector.transform(train)
    test = odetector.transform(test)
train.shape
test.shape


# In[ ]:


# Impute missing values using the median so we will not treat them as outliers later
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = -1, strategy='median')
missing_features = ['android_version', 'size']
imp.fit(train[missing_features])
train.loc[:, missing_features] = imp.transform(train[missing_features])
test.loc[:, missing_features] = imp.transform(test[missing_features])


# ## Starting machine learning
# Split data into predictors and labels, both for train and test

# In[ ]:


def split_samples_labels(df, label_column, keep_features = None):
    Y = df[label_column].values
    X = df.drop(columns = label_column)
    if keep_features is not None:
        X = X[keep_features]
    bins = [Y.min(), np.percentile(Y, 70), Y.max()]
    Y[Y < bins[1]] = 0
    Y[Y >= bins[1]] = 1
    return X, Y


# In[ ]:


x_train, y_train = split_samples_labels(train, 'installs', keep_features=pre_features)
x_test, y_test = split_samples_labels(test, 'installs', keep_features=pre_features)
x_train.shape
x_test.shape


# ## Additional enhancements
# Oversampling and scaling

# In[ ]:


len(y_train[y_train==1])
len(y_train[y_train==0])


# In[ ]:


# the dataset is rather imbalanced, which will skew the results. So we reduce the number of big rating examples
# we can also try upsampling the small rating examples
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
upsample = False
downsample = False
if upsample:
    sampler = RandomOverSampler(random_state = 42)
    x_train, y_train = sampler.fit_resample(x_train,y_train)
elif downsample :
    sampler = RandomUnderSampler(random_state=42)
    x_train, y_train = sampler.fit_resample(x_train, y_train)
len(y_train[y_train==1])
len(y_train[y_train==0])


# In[ ]:


from sklearn import preprocessing
# scale data if needed. forests and trees don't need it. Others do. 
scale = True
if scale:
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)


# ## Tuning models hyperparameters

# In[ ]:


from sklearn import ensemble, tree, svm, neighbors
from sklearn.model_selection import cross_val_score
# KNN - best is n_neighbors = 4
# KNN is very much affected by random upsampling. because many entries are duplicates
knn = neighbors.KNeighborsClassifier()
scores = []
fit_grid = False
if fit_grid:
    try_neighbors = range(1,11)
    for n_neighbors in try_neighbors:
        knn.n_neighbors = n_neighbors;
        val_scores = cross_val_score(knn, x_train, y_train, cv = 3 )
        scores.append(val_scores.mean())
    plt.plot(try_neighbors, scores, 'b*-');
else:
    knn.n_neighbors = 4
    print('Validation score : %.2f' % cross_val_score(knn, x_train, y_train, cv = 3).mean());
    knn.fit(x_train, y_train);
    print('Test score : %.2f' % knn.score(x_test, y_test));


# In[ ]:


# SVC
from sklearn.model_selection import GridSearchCV
svc = svm.SVC(kernel = 'rbf', random_state=42, probability = True,
             C = 1, tol = .001, gamma = 9)
fit_grid = False
if fit_grid:
    params = [{'kernel' : ['rbf'],
            'gamma' : ['scale'],
            'C' : [1],
            'tol' : [0.001],
            'gamma' : [9]}]
    grid = GridSearchCV(estimator = svc,
                  param_grid = params,
                  cv = 3, iid = False)
    grid.fit(x_train, y_train)
    grid.score(x_test, y_test)
else :
    print('Validation score %.2f' % cross_val_score(svc, x_train, y_train, cv = 5).mean())
    svc.fit(x_train, y_train)
    print('Test score %.2f' % svc.score(x_test, y_test))


# In[ ]:


grid.best_estimator_


# In[ ]:


# Random Forest
rf =  ensemble.RandomForestClassifier(n_estimators=30, max_features=2, max_depth=12, min_samples_split=5, random_state=42)
grid_fit = False
if grid_fit:
    params = {
        'max_features' : [2, 4, 5],
        'max_depth' : [12], # 10, 8, 18],
        'min_samples_split' : [5], #, 2, 7, 10],
        'n_estimators': [30, 20, 40]
    }
    grid = GridSearchCV(
        estimator = rf, 
        param_grid = params,
        cv = 5, 
        iid = False, 
        return_train_score=True,
        n_jobs=-1
    )
    grid.fit(x_train, y_train)
    grid.score(x_test, y_test)
    print('Best score: %f' % grid.best_score_)
    print('Best params: %s' % grid.best_params_)
else:
    rf.fit(x_train, y_train)
    rf.score(x_train, y_train)
    rf.score(x_test, y_test)


# In[ ]:


grid.best_estimator_


# In[ ]:


dt = tree.DecisionTreeClassifier(max_depth=9, min_impurity_decrease=0, min_samples_leaf=8, max_features=3, random_state=42)
grid_fit = False
if grid_fit:
    params = {
        'max_features': [1, 2, 3, 4, 5, 6, 7, 8],
        'min_samples_leaf': [2, 3, 4, 5],
        'min_impurity_decrease': [0, 0.01]
    }
    grid = GridSearchCV(
        estimator = dt,
        param_grid = params,
        cv = 5, 
        iid = False, 
        return_train_score=True
    )
    grid.fit(x_train, y_train)
    grid.score(x_test, y_test)
    print('Best score: %f' % grid.best_score_)
    print('Best params: %s' % grid.best_params_)
else:
    dt.fit(x_train, y_train)
    dt.score(x_train, y_train)
    dt.score(x_test, y_test)


# ### Adaboost

# In[ ]:


ada = ensemble.AdaBoostClassifier(
    n_estimators=80,
    base_estimator=None, # decision tree, by default
    random_state = 42
)
grid_fit = False
if grid_fit:
    params = {
        'n_estimators' : [20, 50, 80, 100],
        'learning_rate' : [1, 2, 3, 0.8]
    }
    grid = GridSearchCV(
        estimator = ada,
        param_grid = params,
        cv = 5, 
        iid = False, 
        return_train_score=True,
        n_jobs=-1
    )
    grid.fit(x_train, y_train)
    grid.score(x_test, y_test)
    print('Best score: %f' % grid.best_score_)
    print('Best params: %s' % grid.best_params_)
else:
    ada.fit(x_train, y_train)
    ada.score(x_train, y_train)
    ada.score(x_test, y_test)


# ### Gradient Boosting

# In[ ]:


gbc = ensemble.GradientBoostingClassifier(
    n_estimators=400, 
    loss='deviance', 
    learning_rate=0.15, 
    subsample=0.9, 
    random_state = 42
)
grid_fit = False
if grid_fit:
    params = {
        'n_estimators' : [100, 200, 300, 400],
        'learning_rate' : [0.1, 0.15, 0.2, 0.25],
        'loss': ['deviance', 'exponential'],
        'subsample': [1.0, 0.5, 0.8, 0.85, 0.9, 0.95]
    }
    grid = GridSearchCV(
        estimator = gbc,
        param_grid = params,
        cv = 5,
        iid = False, 
        return_train_score=True, 
        n_jobs=-1
    )
    grid.fit(x_train, y_train)
    grid.score(x_test, y_test)
    print('Best score: %f' % grid.best_score_)
    print('Best params: %s' % grid.best_params_)
else:
    gbc.fit(x_train, y_train)
    gbc.score(x_train, y_train)
    gbc.score(x_test, y_test)


# ### Bagging

# In[ ]:


bag = ensemble.BaggingClassifier(
    base_estimator=None, # decision tree, by default
    n_estimators=300,
    max_samples=100,
    max_features=6,
    random_state = 42
)
grid_fit = False
if grid_fit:
    params = {
        'n_estimators' : [20, 50, 80, 100, 200],
        'max_samples' : [1, 2, 3, 5],
        'max_features': [1, 2, 3, 5]
    }
    grid = GridSearchCV(estimator = bag,
                  param_grid = params,
                  cv = 5, iid = False, return_train_score=True, n_jobs=-1)
    grid.fit(x_train, y_train)
    grid.score(x_test, y_test)
    grid.best_params_
else:
    bag.fit(x_train, y_train)
    bag.score(x_train, y_train)
    bag.score(x_test, y_test)


# ### XGBoost

# In[ ]:


import xgboost
from sklearn.model_selection import GridSearchCV

xgb = xgboost.XGBClassifier(
    n_estimators=320,
    max_depth=3,
    learning_rate=0.1,
    n_jobs=-1,
    gamma=0,
    min_child_weight=1,
#     max_delta_step=0,
    subsample=1,
    random_state = 42
)
grid_fit = False
if grid_fit:
    params = {
        'n_estimators' : [20, 50, 80, 100, 200, 300, 320],
        'max_depth' : [1, 2, 3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'gamma': [0, 1, 2]
    }
    grid = GridSearchCV(estimator = xgb,
                  param_grid = params,
                  cv = 5, iid = False, return_train_score=True, n_jobs=-1)
    grid.fit(x_train, y_train)
    grid.score(x_test, y_test)
    grid.best_params_
else:
    xgb.fit(x_train, y_train)
    xgb.score(x_train, y_train)
    xgb.score(x_test, y_test)


# In[ ]:


from model import Model, ModelsBenchmark
from sklearn.ensemble import VotingClassifier

models = [svc, rf, dt, knn, ada, gbc, bag, xgb]
# add voting method
estimators = [ (model.__class__.__name__,model) for model in models]
voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
models.append(voting_clf)
# benchmark between all models so far
bench = ModelsBenchmark(models);
bench.fit(x_train, y_train)
bench.score(x_train, y_train)
bench.score(x_test, y_test)
bench._scores


# # Evaluating models

# In[ ]:


# reduce dimensionality to be able to plot data
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, random_state = 42);
    
def reduce_dimensions(X, fit=False):
    if fit:
        pca.fit(X)
    return pca.transform(X)


# In[ ]:


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


# Below we make predictions on 2-dimensional data and plot the points labeled wrong.   
# Currently, below part has a bug and doesn't run correctly.

# In[ ]:


# trying to solve the bug below, set the max_features attribute to 2
# rf_plot = rf
# rf_plot.max_features = 2
# dt_plot = dt
# dt_plot.max_features = 2
# vot_plot = voting_clf
# vot_plot.estimators = [svc, knn, rf_plot, dt_plot]
# clf = vot_plot


# In[ ]:


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


# In[ ]:


y_pred == 0


# In[ ]:


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


# In[ ]:


# Visualisation of the decision tree created by the algorithm, for fun and insight
import graphviz
from sklearn.tree import export_graphviz
clf = Model(dt)
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

# In[ ]:


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


# In[ ]:


model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])


# In[ ]:


from keras.utils import to_categorical
no_epochs = 20
batch_size = 1024
history = model.fit(x_train, to_categorical(y_train), validation_split=.3, batch_size = batch_size, epochs = no_epochs )


# In[ ]:


model.evaluate(x_test, to_categorical(y_test))


# In[ ]:


df = pd.DataFrame({'epochs':history.epoch, 'loss': history.history['loss'], 
                   'validation_loss': history.history['val_loss']
                  })
g = sns.pointplot(x="epochs", y="loss", data=df, fit_reg=False, color = 'yellow')
# g = sns.pointplot(x="epochs", y="validation_loss", data=df, fit_reg=False, color='red')


# In[ ]:


import seaborn as sns
df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc']
                   , 'validation_accuracy': history.history['val_acc']
                  })
g = sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
# g = sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')


# # Model voting 
# Maybe below weights will help raise model accuracy

# In[ ]:


from sklearn import ensemble, svm, neighbors
from collections import Counter
estimators = [
#     ('svc', svm.SVC(kernel = 'rbf', random_state=42, probability = True, C = 1, tol = .001, gamma = 9)),
    ('rf', ensemble.RandomForestClassifier(random_state = 42, max_features=3, max_depth=12, n_estimators=30, min_samples_split=5)),
    ('knn', neighbors.KNeighborsClassifier(n_neighbors=2)),
    ('ada', ensemble.AdaBoostClassifier(n_estimators=80, random_state = 42)),
    ('gb', ensemble.GradientBoostingClassifier(n_estimators=400, loss='deviance', learning_rate=0.15, subsample=0.9, random_state = 42))
]
weights = [5, 10, 40, 140]
voting_classifier = ensemble.VotingClassifier(estimators=estimators, voting='soft', weights=weights, n_jobs=-1)
voting_classifier.fit(x_train, y_train)
confidence = voting_classifier.score(x_test, y_test)
predictions = voting_classifier.predict(x_test)
print('accuracy: %s %s' % (confidence, weights))
print('predictions:', Counter(predictions))

