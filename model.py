import collections


class Model:
    def __init__(self, model):
        self.model = model

    def compute_scores(self, train, test):
        self.model.fit(train[0], train[1])
        train_score = self.model.score(train[0], train[1])
        test_score = self.model.score(test[0], test[1])
        return train_score, test_score


class ModelsBenchmark(collections.UserList):
    
    def __init__(self, models=[]):
        super(ModelsBenchmark, self).__init__(models)

    def compute_scores(self, train_data, test_data):
        scores = [(Model(model).compute_scores(train_data, test_data)[1], model) for model in self]
        self._sorted = sorted(scores, key=lambda x: x[0], reverse=True)
        return self._sorted


#TESTING
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn import svm
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    df = pd.read_csv('googleplaystore.csv')
    df_user_reviews = pd.read_csv('googleplaystore_user_reviews.csv')
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
    df_user_reviews.rename(
        columns={'App': 'app_name', 'Translated_Review': 'review', 'Sentiment': 'sentiment',
                 'Sentiment_Polarity': 'polarity', 'Sentiment_Subjectivity': 'subjectivity'}
        , inplace=True)
    df = df[df['installs'] != 'Free']
    new_df = df['installs'].map(lambda s: s[:-1].replace(',', ''))
    new_df[new_df == ''] = 0
    new_df.astype(int).unique()
    df['installs'] = new_df.astype(int)
    df['reviews'] = df['reviews'].astype(int)
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

    Y = dfn['rating'].values
    X = dfn[['size', 'installs', 'reviews', 'last_year_updated', 'name_wc']]



    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    # svr = Model(svm.SVR())
    # train_score, test_score = svr.compute_scores((x_train, y_train), (x_test, y_test))
    # print('Train score : %.4f' % train_score)
    # print('Test score : %.4f' % test_score)
    # print("\n\n\n")


    ##testing

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    models = [svm.SVR(), linear_model.LinearRegression()]
    bench = ModelsBenchmark(models)
    model_scores = bench.get_sorted_scores((x_train, y_train), (x_test, y_test))
    for sc in model_scores:
        print(sc)






