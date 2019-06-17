###Predict wine variety from wine review dataset. The wine review (text description) was embedded by LSTM and he prediction was conducted by Neural Network.

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
import xgboost as xgb
from xgboost import XGBRegressor
import keras.backend as K
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, Flatten, Concatenate, BatchNormalization, Lambda, Bidirectional
import tensorflow as tf
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# %matplotlib inline


def dataLoading(num_variety):

    data_original = pd.read_csv('winemag-data-130k-v2.csv') #Original data loading
    top_variety = data_original.variety.value_counts()[:num_variety].rename_axis('variety').reset_index(name='counts')
    data_selected = data_original[data_original.variety.isin (top_variety['variety'].tolist())] #Selected data with top varieties

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
    def __init__(self):
        pass

    def data_load(self, data):
        data_original = data.copy()
        data = data[['country', 'points', 'province','winery', 'year',  'price']]

        return data, data_original

    def mean_encode(self, data, cols):
        for col in cols:
            folds = KFold(5, random_state=10).split(data)
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

        data.loc[data.country_mean_encoded.isnull(), 'country_mean_encoded'] = data.loc[data.country_mean_encoded.isnull(), 'price']
        data.loc[data.province_mean_encoded.isnull(), 'province_mean_encoded'] = data.loc[data.province_mean_encoded.isnull(), 'country_mean_encoded']
        data.loc[data.province_mean_encoded.isnull(), 'province_mean_encoded'] = data.loc[data.province_mean_encoded.notnull(), 'country_mean_encoded'].mean()

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
        self.random_state= random_state


    def multiclass_label(self, data):
        le = preprocessing.LabelEncoder().fit(data['variety'])
        data['variety'] = le.transform(data['variety'])

        return data

    def non_text_data_selection(self, data):
        ntf_cols = ['country', 'taster_name','province', 'points', 'price', 'year']
        data_ntf = data[ntf_cols]
        data_ntf = pd.get_dummies(data_ntf)

        return data_ntf

    def non_text_data_split(self, data_ntf, data):
        x_train_ntf, x_val_test_ntf, y_train, y_val_test = train_test_split(data_ntf, data.variety, test_size=self.val_test_size,
                                                                            random_state=self.random_state)
        x_val_ntf, x_test_ntf, y_val, y_test = train_test_split(x_val_test_ntf, y_val_test, test_size=self.test_size,
                                                                random_state=self.random_state)

        return x_train_ntf, x_val_ntf, x_test_ntf, y_train, y_val, y_test

    def desc_data_split(self, data, x_train_ntf, x_val_ntf, x_test_ntf):
        x_train_txt = data.loc[x_train_ntf.index, 'description']
        x_val_txt = data.loc[x_val_ntf.index, 'description']
        x_test_txt = data.loc[x_test_ntf.index, 'description']

        return x_train_txt, x_val_txt, x_test_txt

    def desg_data_split(self, data, x_train_ntf, x_val_ntf, x_test_ntf):
        x_train_txt = data.loc[x_train_ntf.index, 'designation']
        x_val_txt = data.loc[x_val_ntf.index, 'designation']
        x_test_txt = data.loc[x_test_ntf.index, 'designation']

        return x_train_txt, x_val_txt, x_test_txt

    def transform(self, data):
        data = self.multiclass_label(data)
        data_ntf = self.non_text_data_selection(data)
        x_train_ntf, x_val_ntf, x_test_ntf, y_train, y_val, y_test = self.non_text_data_split(data_ntf, data)
        x_train_txt1, x_val_txt1, x_test_txt1 = self.desc_data_split(data, x_train_ntf, x_val_ntf, x_test_ntf)
        x_train_txt2, x_val_txt2, x_test_txt2 = self.desg_data_split(data, x_train_ntf, x_val_ntf, x_test_ntf)

        return x_train_ntf, x_val_ntf, x_test_ntf, y_train, y_val, y_test, x_train_txt1, x_val_txt1, x_test_txt1, \
               x_train_txt2, x_val_txt2, x_test_txt2


class TextProcess():
    '''Text preprocess.'''
    def __init__(self, vocabulary_size):
        self.vocabulary_size = vocabulary_size

    def tokenization(self, x_train_txt, x_val_txt, x_test_txt):
        tokenizer = Tokenizer(num_words=self.vocabulary_size)
        tokenizer.fit_on_texts(x_train_txt.values.tolist())
        train_txt_token = tokenizer.texts_to_sequences(x_train_txt.values.tolist())
        val_txt_token = tokenizer.texts_to_sequences(x_val_txt.values.tolist())
        test_txt_token = tokenizer.texts_to_sequences(x_test_txt.values.tolist())

        return train_txt_token, val_txt_token, test_txt_token

    def pad_sequence(self, train_txt_token, val_txt_token, test_txt_token):
        maxlen = max(map(len, train_txt_token))
        train_txt_seq = pad_sequences(train_txt_token, maxlen=maxlen, padding='post', truncating='post')
        val_txt_seq = pad_sequences(val_txt_token, maxlen=maxlen, padding='post', truncating='post')
        test_txt_seq = pad_sequences(test_txt_token, maxlen=maxlen, padding='post', truncating='post')

        return train_txt_seq, val_txt_seq, test_txt_seq, maxlen

    def transform(self, x_train_txt, x_val_txt, x_test_txt):
        train_txt_token, val_txt_token, test_txt_token = self.tokenization(x_train_txt, x_val_txt, x_test_txt)
        train_txt_seq, val_txt_seq, test_txt_seq, maxlen = self.pad_sequence(train_txt_token, val_txt_token, test_txt_token)

        return train_txt_seq, val_txt_seq, test_txt_seq, maxlen


def neuralNetworkModel(embedding_dim1=64, hidden_dim1=64, embedding_dim2=16, hidden_dim2=16, num_class=10):
    text_sequence1 = Input((None,), name='desc_sequence')
    text_sequence2 = Input((None,), name='desg_sequence')
    embedding1 = Embedding(input_dim=vocabulary_size1,
                          output_dim=embedding_dim1,
                          input_length=maxlen1,
                          name='desc_embedding')
    embedding2 = Embedding(input_dim=vocabulary_size2,
                          output_dim=embedding_dim2,
                          input_length=maxlen2,
                          name='desg_embedding')
    text_embedded1 = embedding1(text_sequence1)
    text_embedded2 = embedding2(text_sequence2)
    lstm_desc = Bidirectional(LSTM(units=hidden_dim1, dropout=0.2, recurrent_dropout=0.2))(text_embedded1)
    lstm_desg = Bidirectional(LSTM(units=hidden_dim2, dropout=0.2, recurrent_dropout=0.2))(text_embedded2)

    non_text_inputs = Input((x_train_ntf.shape[1],), name='non_text_features')
    non_text_features = BatchNormalization()(non_text_inputs)

    X = Concatenate()([lstm_desc, lstm_desg, non_text_features])
    X = BatchNormalization()(X)
    # X = Dense(2048, activation='relu')(X)
    X = Dense(1024, activation='relu')(X)
    X = Dense(512, activation='relu')(X)
    X = Dense(256, activation='relu')(X)
    X = Dense(128, activation='relu')(X)
    X = Dense(64, activation='relu')(X)
    # X = BatchNormalization()(X)
    X = Dense(32, activation='relu')(X)
    # X = BatchNormalization()(X)
    X = Dense(16, activation='relu')(X)
    # X = BatchNormalization()(X)
    output = Dense(num_class, activation='softmax')(X)
    model = Model(inputs=[text_sequence1, text_sequence2, non_text_inputs], outputs=output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':

    '''Part 1. Data processing.'''
    num_variety = 10
    data = dataLoading(num_variety)
    data = Preprocess().transform(data)
    mse, data = PriceImputer().transform(data)
    data.designation = data.designation.astype('str')
    x_train_ntf, x_val_ntf, x_test_ntf, y_train, y_val, y_test, x_train_txt1, x_val_txt1, x_test_txt1, \
    x_train_txt2, x_val_txt2, x_test_txt2 = DataSplit(0.3, 0.5, 10).transform(data)
    train_txt_seq1, val_txt_seq1, test_txt_seq1, maxlen1 = TextProcess(1000).transform(x_train_txt1, x_val_txt1, x_test_txt1)
    train_txt_seq2, val_txt_seq2, test_txt_seq2, maxlen2 = TextProcess(10).transform(x_train_txt2, x_val_txt2, x_test_txt2)

    '''Part 2. Modeling.'''
    vocabulary_size1 = 1000
    vocabulary_size2 = 10
    K.clear_session()
    model = neuralNetworkModel(num_class=num_variety)
    model.summary()

    es = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    checkpoint = ModelCheckpoint("variety_model.model", monitor='val_loss', verbose=1, save_best_only=True)

    model.fit(x=[train_txt_seq1,train_txt_seq2, x_train_ntf],
              y=to_categorical(y_train),
              validation_data=([val_txt_seq1,val_txt_seq2, x_val_ntf], to_categorical(y_val)),
              batch_size=128,
              epochs=100,
              shuffle=True,
              callbacks = [es, checkpoint])
    # model = load_model("variety_model.model")
    y_pred = model.predict([test_txt_seq1, test_txt_seq2, x_test_ntf])
    y_pred = np.argmax(y_pred, axis=1)

    print (accuracy_score(y_test, y_pred))
    print (classification_report(y_test, y_pred))