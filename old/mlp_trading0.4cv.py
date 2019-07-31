'''Cross validation

bug tracker
-----------------------------
priority | name
-----------------------------
1        | rubi, how accuracy so high?? 64%!
2        | should use PURGED K-FOLD Cross Validation or  TimeSeriesSplit instead of standart split
3        | rubi, which normalize function to use?
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from data_splitter.splitter_k_fold_purged import PurgedKFoldDataSplitter
from build_model.utils import *

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

# import pandas.io.data as web

# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk

batch_size = 128  # we cannot pass the entire data into network at once , so we divide it to batches . number of samples that we will pass through the network at 1 time and use for each epoch. default is 32
epochs = 100  # 50  #  iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
# iterations  = 60000/128
# size_input = 10  # x_train.shape[1] # no of features
size_hidden = 512  # If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to overfit
size_output = 2  # there are 3 classes (buy.sell. hold) or (green,red,hold)
name_output = ['Green bar', 'Red Bar']  # 'no direction Bar'
# iterations  = 60000/128
symbol = '^GSPC'  # ^GSPC=SP500 (3600->1970 or 12500 =2000) DJI(300, 1988)  QQQ(300, 2000) GOOG XLF XLV
skipDays = 3600  # 12500 total 13894 daily bars
percentTestSplit = 0.33  # 33% from data will go to test

print('\nLoading  data')
print('\n======================================')
# Define date range
# start_date, end_date='1970-01-03','2019-07-12'
# dates=pd.date_range(start_date,end_date)
# print("dates="  ,dates)
# print("date[0]=",dates[0])
#
# # Define stock symbols
# symbols = []#'TSLA', 'GOOG', 'FB']  # SPY will be added in get_data()
# import get_prices as hist
# hist.get_stock_data(symbol, start_date=start_date, end_date=end_date)
# process = DataProcessing("stock_prices.csv", 0.9)
# process.gen_test(10)
# process.gen_train(10)
#

# Get stock data
df_all = get_data_from_disc(symbol, skipDays)
print(df_all.tail())

# Slice and plot
# plot_selected(df_all, [  'Close', 'sma200'], shouldNormalize=True, symbol=symbol)

# Slice and plot
plot_selected(df_all, title='TA-price of ' + symbol + ' vs time', columns=['Close', 'sma200'], shouldNormalize=False,
              symbol=symbol)

plot_selected(df_all.tail(500), title='TA-sma 1,10,20,50,200 of ' + symbol + ' vs time',
              columns=['Close', 'sma10', 'sma20', 'sma50', 'sma200', 'sma400', 'bb_hi10', 'bb_lo10', 'bb_hi20',
                       'bb_lo20', 'bb_hi50', 'bb_lo200', 'bb_lo50', 'bb_hi200'], shouldNormalize=False, symbol=symbol)

plot_selected(df_all.tail(500), title='TA-range sma,bband of ' + symbol + ' vs time',
              columns=['range_sma', 'range_sma1', 'range_sma2', 'range_sma3', 'range_sma4', 'rel_bol_hi10',
                       'rel_bol_hi20', 'rel_bol_hi200', 'rel_bol_hi50'], shouldNormalize=False, symbol=symbol)
plot_selected(df_all.tail(500), title='TA-rsi,stoc of ' + symbol + ' vs time',
              columns=['rsi10', 'rsi20', 'rsi50', 'rsi200', 'stoc10', 'stoc20', 'stoc50', 'stoc200'],
              shouldNormalize=False, symbol=symbol)

# plot_selected(df, ['Date','Close']                                    , start_date, end_date, shouldNormalize=False)
elements = df_all.size
shape = df_all.shape

print('\nInput Data')
print('\n======================================')
df_data = df_all.loc[:,
          ['sma10', 'sma20', 'sma50', 'sma200', 'sma400', 'range_sma', 'range_sma1', 'range_sma2', 'range_sma3',
           'range_sma4', 'bb_hi10', 'bb_lo10', 'bb_hi20', 'bb_lo20', 'bb_hi50', 'bb_lo50', 'bb_hi200', 'bb_lo200',
           'rel_bol_hi10', 'rel_bol_lo10', 'rel_bol_hi20', 'rel_bol_lo20', 'rel_bol_hi50', 'rel_bol_lo50',
           'rel_bol_hi200', 'rel_bol_lo200', 'rsi10', 'rsi20', 'rsi50', 'rsi200', 'stoc10', 'stoc20', 'stoc50',
           'stoc200']]
print('columns=', df_data.columns)
print('\ndata describe=\n', df_data.describe())
print('shape=', str(shape), " elements=" + str(elements), ' rows=', str(shape[0]))

print('\nOutput Data (Labeling)')
print('\n======================================')
df_y = df_all['isUp']  # np.random.randint(0,2,size=(shape[0], ))
# y = np.random.randint(0,2,size=(shape[0], ))
# print(y)
# df_y = pd.DataFrame()#, columns=list('is_up'))
# df_y['isUp'] = y
print(df_y)

ts_cv = PurgedKFoldDataSplitter    (n_splits=5, gap_percentage=2.5)
#ts_cv = StratifiedKFold(n_splits=5, shuffle=False)

# ts_cv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for train_indices, test_indices in ts_cv.split(df_data.values):
    x_train = df_data.values[train_indices]
    x_test = df_data.values[test_indices]
    print('x_train shape=', str(x_train.shape))
    print('x_test shape=', str(x_test.shape))

    print('Observations: %d' % (len(x_train) + len(x_test)))
    print('Training Observations: %d' % (len(x_train)))
    print('Testing Observations: %d' % (len(x_test)))

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    print('\ntrain data', x_train.shape)
    print('\ntest data', x_test.shape)

    y_train = df_y.values[train_indices]
    y_test = df_y.values[test_indices]
    print('y_train shape=', str(y_train.shape))
    print('y_test shape=', str(y_test.shape))

    y_train = keras.utils.to_categorical(y_train, size_output)
    y_test = keras.utils.to_categorical(y_test, size_output)
    print('\ntrain labels', y_train.shape)
    print('\ntest labels', y_test.shape)

    size_input = x_train.shape[1]  # no of features

    # create model
    model = Sequential()  # stack of layers
    # model.add(tf.keras.layers.Flatten())
    model.add(Dense(size_hidden, activation='relu', input_shape=(size_input,)))
    model.add(Dropout(0.2))  # for generalization
    model.add(Dense(size_hidden, activation='relu'))
    model.add(Dropout(0.2))  # regularization technic by removing some nodes
    model.add(Dense(size_output,
                    activation='softmax'))  # last layer always has softmax(except for regession problems and  binary- 2 classes where sigmoid is enough)
    # For binary classification, softmax & sigmoid should give the same results, because softmax is a generalization of sigmoid for a larger number of classes.
    # softmax:  loss: 0.3099 - acc: 0.8489 - val_loss: 0.2929 - val_acc: 0.8249
    # sigmoid:  loss: 0.2999 - acc: 0.8482 - val_loss: 0.1671 - val_acc: 0.9863

    # Prints a string summary of the  neural network.')
    model.summary()

    # Compile model
    model.compile(loss='categorical_crossentropy',  # measure how accurate the model during training
                  optimizer=RMSprop(),  # this is how model is updated based on data and loss function
                  metrics=['accuracy'])

    # Fit the model
    # model.fit(df_data[train], df_y[train], epochs=150, batch_size=10, verbose=0)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # history = model.fit(x_train
    #                     , y_train
    #                     , batch_size=batch_size
    #                     , epochs=epochs
    #                     , validation_data=(x_test, y_test)
    #                     , verbose=1)

    # evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("----end of itteration: %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cv_scores.append(scores[1] * 100)

print("-------total accuracy : %.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
