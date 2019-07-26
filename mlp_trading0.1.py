'''Trains a simple deep NN MLP (Multilayer perceptron) on the MNIST dataset. 28X28 images of digits

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.


3 Techniques to Prevent Overfitting
1. Early Stopping: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.
2. Image Augmentation: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.
3. neoron Dropout: Removing a random selection of a fixed number of neurons in a neural network during training.

'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import random
#import pandas.io.data as web
import os
from sklearn.model_selection import train_test_split
# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk
# print("creating deep learning to classify images to digit(0-9). MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8")
# These MNIST images of 28×28 (=784 features) pixels are represented as an array of numbers  whose values range from [0, 255] (0=white,255=black) of type uint8.



def kpi_returns(prices):
    return ((prices-prices.shift(-1))/prices)[:-1]



def kpi_sharpeRatio():

    risk_free_rate = 2.25 # 10 year US-treasury rate (annual) or 0
    sharpe = 2
    #  ((mean_daily_returns[stocks[0]] * 100 * 252) -  risk_free_rate ) / (std[stocks[0]] * 100 * np.sqrt(252))
    return sharpe

def kpi_commulativeReturn():
    return 2


def kpi_risk(df):
    return df.std()


def kpi_sharpeRatio():
    return 2



def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def loss_log():
    return 2

def loss_mse():
    return 2


def loss_gdc():
    return 2

def activation_sigmoid():
    return 2


def plot_selected(df, columns, start_index, end_index, shouldNormalize = True):
    """Plot the desired columns over index values in the given range."""
    #df = df[columns][start_index:end_index]
    df.loc[start_index:end_index, columns]
    if shouldNormalize:
        df = normalize(df)
    plot_data(df, title='stock price (normalized)')





def plot_data(df, title="stock price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def plot_image(df, title):
    plt.figure()
    plt.imshow(df[0])#, cmap=plt.cm.binary)
    plt.colorbar()
    plt.gca().grid(False)
    plt.title(title)
    plt.show()

def plot_images(x,y, title):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=plt.cm.binary)
        plt.xlabel(y[i])
    plt.show()

def plot_stat_loss_vs_time(history_dict) :
    acc      = history_dict['acc']
    val_acc  = history_dict['val_acc']
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss   , 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss over time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_stat_accuracy_vs_time(history_dict) :
    acc      = history_dict['acc']
    val_acc  = history_dict['val_acc']
    loss     = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc    , 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b' , label='Validation acc')
    plt.title('Training and validation accuracy over time')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def plot_stat_train_vs_test(history):
    hist = history.history
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.plot(hist['loss'])#train loss
    plt.plot(hist['val_loss'])#validation loss
    plt.title    ('model loss')
    plt.legend   (['train Error', 'test Error'], loc='upper right')
    plt.show()

# normalize to first row
def normalize(df):
    return df/df.iloc[0,:]


def normalize2(x):
    train_stats = x_train.describe()
    return (x - train_stats['mean']) / train_stats['std']


def symbol_to_path(symbol, base_dir=""):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data_from_disc(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'GOOG' not in symbols:  # add GOOG for reference, if absent
        symbols.insert(0, 'GOOG')

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])

        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        print(df_temp.head())
        df = df.join(df_temp)
        if symbol == 'GOOG':  # drop dates GOOG did not trade
            df = df.dropna(subset=["GOOG"])

    return df

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files.
    https://finance.yahoo.com/quote/%5EGSPC/history?period1=-630986400&period2=1563138000&interval=1d&filter=history&frequency=1d
    """

    df = pd.DataFrame(index=dates)
    if 'SP500' not in symbols:  # add GOOG for reference, if absent
        symbols.insert(0, 'SP500')

    for symbol in symbols:
        df_temp = pd.read_csv(  symbol_to_path(symbol)
                              , index_col  = 'Date'
                              , parse_dates= True
                              , usecols    = ['Date', 'Adj Close']
                              , na_values  = ['nan'])

        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        print(df_temp.head())
        sma10 = df_temp.rolling(window=10).mean()
        sma20 = df_temp.rolling(window=20).mean()
        print ('sma_20=',sma20)
        print ('sma_10=',sma10)
        sma10 = sma10.rename(columns={symbol: symbol+'sma10'})
        sma20 = sma20.rename(columns={symbol: symbol+'sma20'})
        df = df.join(df_temp)\
               .join(sma10)\
               .join(sma20)
        if symbol == 'SP500':
            df = df.dropna(subset=["SP500"])# drop dates GOOG did not trade

    return df


def get_data_from_web(symbol):
    start, end = '1950-01-03','2019-07-12'#'2007-05-02', '2016-04-11'
    data = web.DataReader(symbol, 'yahoo', start, end)
    data=pd.DataFrame(data)
    prices=data['Adj Close']
    prices=prices.astype(float)
    return prices


def get_state(parameters, t, window_size = 20):
    outside = []
    d = t - window_size + 1
    for parameter in parameters:
        block = (
            parameter[d : t + 1]
            if d >= 0
            else -d * [parameter[0]] + parameter[0 : t + 1]
        )
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        for i in range(1, window_size, 1):
            res.append(block[i] - block[0])
        outside.append(res)
    return np.array(outside).reshape((1, -1))


def kpi_sharpeRatio():
    return 2









print('Loading train & test data')
# Define date range
start_date, end_date='1950-01-03','2019-07-12'
dates=pd.date_range(start_date,end_date)
print("dates="  ,dates)
print("date[0]=",dates[0])

# Define stock symbols
symbols = []#'TSLA', 'GOOG', 'FB']  # SPY will be added in get_data()

# Get stock data
df = get_data(symbols, dates)
print(df.tail())
print(df.describe())

# Slice and plot
plot_selected(df, symbols, start_date, end_date, shouldNormalize=True)

# Slice and plot
plot_selected(df, symbols, start_date, end_date, shouldNormalize=False)
elements = df.size
shape=df.shape
print('shape=',str(shape), " elements="+str(elements), ' rows=',str(shape[0]))
(x_train, x_test)  = train_test_split(df.values, test_size=0.33, shuffle=False)


print('Labeling data')
y = np.random.randint(0,2,size=(shape[0], ))
#print(y)
df_y = pd.DataFrame()#, columns=list('is_up'))
df_y['isUp'] = y
#print(df_y)
(y_train, y_test)  = train_test_split(y, test_size=0.33, shuffle=False)
print(df_y.tail())
print(df_y.describe())
#(x_train, y_train)  = train_test_split(df.as_matrix(), test_size=0.33, shuffle=False)


print('\ntrain data', x_train.shape)
print(x_train[0])
print(x_train[1])

#plot_image(x_train, 'picture example')
#plot_images(x_train, y_train, 'picture examples')
print('\ntest data', x_test.shape)
print(x_test[0])
print(x_test[1])


print('\ntrain labels',y_train.shape)
print(y_train[0])
print(y_train[1])


print('\ntest labels', y_test.shape)
print(y_test[0])
print(y_test[1])

print(x_train.shape[0], 'train samples')
print( x_test.shape[0], 'test samples')
#df_GOOG = pd.read_csv('GOOG.csv', index_col="Date", parse_dates=True, usecols=["Date","Close","High","Low"], na_values=["nan"])

# df.GOOG.add(avg (df_GOOG))
# df.GOOG.add(atr (df_GOOG))
# df.GOOG.add(band(df_GOOG))
# double currAtr  = iATR(aSymbol, aTf,  aPeriod,  aShift);
# double currAtr  = iCustom(aSymbol, aTf, "AvgAtr3",aPeriod,   10,0,0,   20,0,0, 0,aShift);
# double avgAtr   = iCustom(aSymbol, aTf, "AvgAtr3",aPeriod,   10,0,0,   20,0,0, 1,aShift);
# double avgAtr2  = iCustom(aSymbol, aTf, "AvgAtr3",aPeriod,   10,0,0,   20,0,0, 2,aShift);
#currSpread
#currSwap
#economic
# (x_train, y_train), (x_test, y_test) = train_test_split(X, y_train, test_size=0.33



batch_size  = 128# we cannot pass the entire data into network at once , so we divide it to batches . number of samples that we will pass through the network at 1 time and use for each epoch. default is 32
epochs      = 12 #  iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
#iterations  = 60000/128
num_input   = x_train.shape[1] # features
num_hidden  = 512 # If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to overfit
num_classes = 2 # there are 3 classes (buy.sell. hold) or (green,red,hold)

#iterations  = 60000/128


print('\nClean data)')
#dataset.isna().sum()
#dataset = dataset.dropna()

print('\nNormalize   to    0-1 (float)')
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test , axis=1)
print(x_train[0])
#print(x_train2[0])
#plot_image(x_test,'picture example')

print('\nTransform data. Convert class vectors to binary class matrices (for ex. convert digit 7 to bit array[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test  = keras.utils.to_categorical( y_test, num_classes)
print('y_train[0]=', y_train[0])
print('y_test [0]=',  y_test[0])

print('\ncreate model...')
model = Sequential()# stack of layers
#model.add(tf.keras.layers.Flatten())
model.add(Dense  (num_hidden, activation='relu', input_shape=(num_input,)))
model.add(Dropout(0.2))
model.add(Dense  (num_hidden, activation='relu'))
model.add(Dropout(0.2))#regularization technic by removing some nodes
model.add(Dense  (num_classes, activation='softmax'))# last layer always has softmax(except for regession problems and  binary- 2 classes where sigmoid is enough)
# Prints a string summary of the  neural network.')
model.summary()
model.compile(loss      = 'categorical_crossentropy',# measure how accurate the model during training
              optimizer = RMSprop(),#this is how model is updated based on data and loss function
              metrics   = ['accuracy'])



print('\ntrain model...')
history = model.fit(  x_train
                    , y_train
                    , batch_size     = batch_size
                    , epochs         = epochs
                    , validation_data= (x_test, y_test)
                    , verbose        = 1
                  # , callbacks=[early_stop, PrintDot()]#Early stopping is a useful technique to prevent overfitting.
                      )

print('\nplot_accuracy_loss_vs_time...')
history_dict = history.history
print(history_dict.keys())
plot_stat_loss_vs_time     (history_dict)
plot_stat_accuracy_vs_time (history_dict)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
plot_stat_train_vs_test(history)

print('\nEvaluate the model with unseen data. pls validate that test accuracy =~ train accuracy and close to 1.0')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:'    , score[0], ' (is it close to 0?)')#Test loss: 0.089
print('Test accuracy:', score[1], ' (is it close to 1 and close to train accuracy?)')#Test accuracy here: 0.9825, Test accuracy in mnist_cnn: 0.9903

print('\nPredict unseen data with 10 probabilities for 10 classes(choose the highest)')
predictions = model.predict(x_test)
print('labeled   as ' , y_test[0]    , ' highest confidence for ' , np.argmax(y_test[0]))
print('predicted as ' ,predictions[0], ' highest confidence for ' , np.argmax(predictions[0]))

filename='mnist_mlp.model'
print('\nSave model as ',filename)
model.save(filename)# 5.4 mb
newModel = tf.keras.models.load_model(filename)

