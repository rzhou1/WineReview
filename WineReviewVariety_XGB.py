###Predict wine variety from wine review dataset. The wine review (text description) was embedded by TFIDF and the prediction was conducted by XGBoost.

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


# %matplotlib inline


def dataLoading(num_variety):
    data_original = pd.read_csv('winemag-data-130k-v2.csv')  # Original data loading
    top_variety = data_original.variety.value_counts()[:num_variety].rename_axis('variety').reset_index(name='counts')
    data_selected = data_original[data_original.variety.isin(top_variety['variety'].tolist())]  # Selected data with top varieties

    return data_selected


class Preprocess():
    def __init__(self):
        pass

    def drop_duplicates(self, data, features):
        """Drop duplicates in description."""
        for feature in features:
            data = data.drop_duplicates(feature)
        return data

    def remove_winery_from_title(self, x):
        title, winery = x
        subtitle = title.replace(winery, '').strip()
        return subtitle

    def extract_year(self, data):
        data['year'] = data[['title', 'winery']].apply(self.remove_winery_from_title, axis=1).str.extract(
            r'(\d{4})').astype('float')
        data.loc[data.year < 1900, 'year'] = 'nan'
        data.loc[data.year.isnull(), 'year'] = 0
        data['year'] = data['year'].astype('float')

        return data

    def log_transform(self, data, features):
        for feature in features:
            data[feature] = np.log1p(data[feature])
        return data

    def imputer_variety(self, data):
        """Imputer missing values of variety by using values from the same winery and the same
        year."""
        non_variety_wineries = data.loc[data.variety.isnull(), 'winery'].tolist()
        non_variety_years = data.loc[data.variety.isnull(), 'year'].tolist()

        for i in range(len(non_variety_wineries)):
            variety_to_imputer = list(set(data.loc[(data.winery == non_variety_wineries[i]) &
                                                   (data.year == non_variety_years[i]) &
                                                   (data.variety.notnull()), 'variety'].tolist()))
            data.loc[(data.winery == non_variety_wineries[i]) & (data.year == non_variety_years[i]) &
                     (data.variety.isnull()), 'variety'] = variety_to_imputer

        return data

    def imputer_locations_by_winery(self, data, features):
        """Imputer missing values of country and province by using values from the same winery."""
        for feature in features:
            non_location_winery = {}
            wineries_non_feature = data.loc[data[feature].isnull(), 'winery'].tolist()
            for w in wineries_non_feature:
                if w not in non_location_winery:
                    non_location_winery[w] = list(set(data.loc[(data.winery == w) & (data[feature].notnull()),
                                                               feature].tolist()))
            non_location_winery = {k: v for k, v in non_location_winery.items() if v}
            for k, v in non_location_winery.items():
                data.loc[(data['winery'] == k) & (data[feature].isnull()), feature] = v[0]

        return data

    def imputer_w_unknown(self, data, features):
        """Assigned 'unknown' to those missing values of 'country', 'province', and'taste_name' that cannot be
        imputered by winery etc.."""
        for feature in features:
            data.loc[data[feature].isnull(), feature] = 'unknown'
        return data

    def reset_index(self, data):
        data = data.reset_index()
        data = data.drop(['index', 'Unnamed: 0'], axis=1)

        return data

    def transform(self, data):

        data = self.drop_duplicates(data, ['description'])
        data = self.extract_year(data)
        data = self.imputer_variety(data)
        data = self.imputer_locations_by_winery(data, ['country', 'province'])
        data = self.imputer_w_unknown(data, ['country', 'province', 'taster_name'])
        data = self.reset_index(data)

        return data


class PriceImputer():
    '''Imputer missing values in price for dataset Wine Review by XGBRegressor.'''

    def __init__(self, random_state):
        self.random_state = random_state

    def data_load(self, data):
        data_original = data.copy()
        data = data[['country', 'points', 'province', 'winery', 'year', 'price']]

        return data, data_original

    def mean_encode(self, data, cols):
        for col in cols:
            folds = KFold(5, random_state=self.random_state).split(data)
            means = np.array([])
            for train_index, test_index in folds:
                X_train, X_test = data.iloc[train_index, :], data.iloc[test_index, :]
                mean = X_test[col].map(X_train.groupby(col)['price'].mean())
                means = np.concatenate([means, mean], axis=-1)
            data['{}_mean_encoded'.format(col)] = means

        return data

    def imputer_encoded_means(self, data):

        '''Missing country_encoded mean: 2 values, only one sample for each country, equals to price.
        Similarly for province_encoded_mean: equal to price or mean of all values.'''

        data.loc[data.country_mean_encoded.isnull(), 'country_mean_encoded'] = data.loc[
            data.country_mean_encoded.isnull(), 'price']
        data.loc[data.province_mean_encoded.isnull(), 'province_mean_encoded'] = data.loc[
            data.province_mean_encoded.isnull(), 'country_mean_encoded']
        data.loc[data.province_mean_encoded.isnull(), 'province_mean_encoded'] = data.loc[
            data.province_mean_encoded.notnull(), 'country_mean_encoded'].mean()

        return data

    def data_split(self, data):
        data = data.drop(['country', 'province', 'winery', 'winery_mean_encoded'], axis=1)

        columns = data.columns.tolist()
        columns = columns[:2] + columns[3:] + columns[2:3]
        data = data[columns]

        train, val = train_test_split(data.loc[data.price.notnull(), :], test_size=0.3, random_state=10)
        test = data.loc[data.price.isnull(), :]

        return train, val, test

    def model_xgboost(self, train, val, test):
        model = XGBRegressor(objective='reg:linear', learning_rate=0.03, max_depth=3, min_child_weight=3, subsample=0.7,
                             n_estimators=500, n_jobs=4)
        model.fit(train.iloc[:, :-1], np.log(train.iloc[:, -1]))
        y_hat = model.predict(val.iloc[:, :-1])
        y_pred = model.predict(test.iloc[:, :-1])
        test.loc[:, 'price'] = np.exp(y_pred)

        return mean_squared_error(y_hat, np.log(val.iloc[:, -1])), test

    def price_imputer(self, train, val, test, data_original):
        data_imputer = pd.concat([train, val, test])
        data_original.loc[:, 'price'] = data_imputer.loc[:, 'price']

        return data_original

    def transform(self, data):
        data, data_original = self.data_load(data)
        data = self.mean_encode(data, ['country', 'province', 'winery'])
        data = self.imputer_encoded_means(data)
        train, val, test = self.data_split(data)
        mse, test = self.model_xgboost(train, val, test)
        data_imputer = self.price_imputer(train, val, test, data_original)

        return mse, data_imputer




class DataSplit():

    def __init__(self, val_test_size, test_size, random_state):
        self.val_test_size = val_test_size
        self.test_size = test_size
        self.random_state = random_state

    def multiclass_label(self, data):
        le = preprocessing.LabelEncoder().fit(data['variety'])
        data['variety'] = le.transform(data['variety'])

        return data

    def one_hot_encode(self, data):
        cols_drop = ['description', 'designation', 'region_1', 'region_2', 'taster_twitter_handle', 'title', 'winery']
        data = data.drop(cols_drop, axis=1)
        data = pd.get_dummies(data)

        return data

    def data_split(self, data):
        cols = data.columns.tolist()
        cols = cols[2:3] + cols[0:2] + cols[3:]
        data = data[cols]

        data = data.loc[data.variety.isin([i for i in range(10)])]

        X = data.iloc[:, 1:]
        y = data.variety

        X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=self.val_test_size, random_state=self.random_state)

        X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=self.test_size, random_state=self.random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def transform(self, data):
        data = self.multiclass_label(data)
        data = self.one_hot_encode(data)
        X_train, X_val, X_test, y_train, y_val, y_test = self.data_split(data)

        return X_train, X_val, X_test, y_train, y_val, y_test

def text_process(X_train, X_val, X_test, max_features):
    punctuations = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', "%"]
    stop_words = text.ENGLISH_STOP_WORDS.union(punctuations)

    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

    def tokenize(text):
        return [stemmer.stem(word) for word in tokenizer.tokenize(text.lower())]

    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize, max_features=max_features)
    vectorizer = vectorizer.fit(X_train.values)
    X_train_txt = vectorizer.transform(X_train.values)
    X_val_txt = vectorizer.transform(X_val.values)
    X_test_txt = vectorizer.transform(X_test.values)

    X_train_txt = X_train_txt.toarray()
    # X_train_txt = pd.DataFrame(X_train_txt)
    X_val_txt = X_val_txt.toarray()
    # X_val_txt = pd.DataFrame(X_val_txt)
    X_test_txt = X_test_txt.toarray()
    # X_test_txt = pd.DataFrame(X_test_txt)

    return X_train_txt, X_val_txt, X_test_txt


def modelXGBoost(X_train, y_train, X_val, X_test, num_variety):
    xgbc = xgb.XGBClassifier(max_depth=3,
                             learning_rate=0.1,
                             n_estimators=500,
                             objective='multi:softmax',
                             num_class=num_variety,
                             booster='gbtree',
                             random_state=10,
                             n_jobs=4)

    xgbc.fit(X_train, y_train)

    y_hat_val = xgbc.predict(X_val)
    y_hat_test = xgbc.predict(X_test)

    return y_hat_val, y_hat_test



if __name__ == '__main__':
    num_variety = 10
    data = dataLoading(num_variety)
    data = Preprocess().transform(data)
    mse, data = PriceImputer(100).transform(data)
    data.designation = data.designation.astype('str')

    X_train, X_val, X_test, y_train, y_val, y_test = DataSplit(0.3,0.5,0).transform(data)

    X_train_desc = data.loc[X_train.index, 'description']
    X_val_desc = data.loc[X_val.index, 'description']
    X_test_desc = data.loc[X_test.index, 'description']
    X_train_desc, X_val_desc, X_test_desc = text_process(X_train_desc, X_val_desc, X_test_desc, 500)

    X_train_desg = data.loc[X_train.index, 'designation']
    X_val_desg = data.loc[X_val.index, 'designation']
    X_test_desg = data.loc[X_test.index, 'designation']
    X_train_desg, X_val_desg, X_test_desg = text_process(X_train_desg, X_val_desg, X_test_desg, 10)

    X_train_np = X_train.as_matrix()
    X_val_np = X_val.as_matrix()
    X_test_np = X_val.as_matrix()

    X_train_npx = np.concatenate((X_train_np, X_train_desc, X_train_desg), axis=1)
    X_val_npx = np.concatenate((X_val_np, X_val_desc, X_val_desg), axis=1)
    X_test_npx = np.concatenate((X_test_np, X_test_desc, X_val_desg), axis=1)

    y_hat_val, y_hat_test = modelXGBoost(X_train_npx, y_train, X_val_npx, X_test_npx, num_variety)

    print "Metrics for validation data:\n"
    print classification_report(y_val, y_hat_val)
    print accuracy_score(y_val, y_hat_val)

    print "Metrics for test data:\n"
    print classification_report(y_test, y_hat_test)
    print accuracy_score(y_test, y_hat_test)


