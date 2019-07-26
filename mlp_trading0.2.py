'''Trains a simple deep NN MLP (Multilayer perceptron) on the SP500

Gets to 90.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
1 seconds per epoch on a K520 GPU.

3 Techniques to Prevent Overfitting
1. Early Stopping: In this method, we track the loss on the validation set during the training phase and use it to determine when to stop training such that the model is accurate but not overfitting.
2. Image Augmentation: Artificially boosting the number of images in our training set by applying random image transformations to the existing images in the training set.
3. neoron Dropout: Removing a random selection of a fixed number of neurons in a neural network during training. model.add(Dropout(0.2, input_shape=(60,)))
4. L1, L2 Regularization:https://classroom.udacity.com/courses/ud188/lessons/b4ca7aaa-b346-43b1-ae7d-20d27b2eab65/concepts/75935645-a408-4685-bd9c-5f234e1b0761


bug tracker
-----------------------------
priority | name
-----------------------------
1        | rubi, how accuracy so high??
2        | rubi, which normalize function to use?
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from sklearn.model_selection import train_test_split

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
#import pandas.io.data as web

# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk

batch_size  = 128# we cannot pass the entire data into network at once , so we divide it to batches . number of samples that we will pass through the network at 1 time and use for each epoch. default is 32
epochs      = 50 #  iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
#iterations  = 60000/128
size_input   = 10#x_train.shape[1] # no of features
size_hidden  = 512 # If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to overfit
size_output  = 2 # there are 3 classes (buy.sell. hold) or (green,red,hold)
name_output  = ['Green bar', 'Red Bar']# 'no direction Bar'
#iterations  = 60000/128
symbol       = '^GSPC' #SP500 (3600) DJI(300) GOOG XLF XLV QQQ


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
df_all = get_data_from_disc(symbol, 3600)
print(df_all.tail())

# Slice and plot
#plot_selected(df_all, [  'Close', 'sma200'], shouldNormalize=True, symbol=symbol)

# Slice and plot
plot_selected(df_all, [ 'Close',  'sma200'],  shouldNormalize=False, symbol=symbol)
#plot_selected(df, ['Date','Close']                                    , start_date, end_date, shouldNormalize=False)
elements = df_all.size
shape=df_all.shape



print('\nsplit to train & test data')
print('\n======================================')
#df_data = df_all.loc[:,   [ 'Open', 'High', 'Low', 'Close', 'sma10', 'sma20', 'sma50',  'sma200', 'range', 'range_sma']]
df_data = df_all.loc[:,   [  'sma10', 'sma20', 'sma50',  'sma200', 'range_sma']]#range, ohlc can give the classifier large clue
print('\ndata describe=\n',df_data.describe())
print('shape=',str(shape), " elements="+str(elements), ' rows=',str(shape[0]))
(x_train, x_test)  = train_test_split(df_data.values, test_size=0.33, shuffle=False)

print('\ntrain data', x_train.shape)
print(x_train[0])
print(x_train[1])

#plot_image(x_train, 'picture example')
#plot_images(x_train, y_train, 'picture examples')
print('\ntest data', x_test.shape)
print(x_test[0])
print(x_test[1])


print('\nLabeling')
print('\n======================================')
df_y = df_all['isUp']#np.random.randint(0,2,size=(shape[0], ))
#y = np.random.randint(0,2,size=(shape[0], ))
#print(y)
#df_y = pd.DataFrame()#, columns=list('is_up'))
#df_y['isUp'] = y
print(df_y)
(y_train, y_test)  = train_test_split(df_y.values, test_size=0.33, shuffle=False)
print(df_y.tail())
print('\nlabel describe\n',df_y.describe())
#(x_train, y_train)  = train_test_split(df.as_matrix(), test_size=0.33, shuffle=False)

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





print('\nClean data)')
print('\n======================================')
#dataset.isna().sum()
#dataset = dataset.dropna()

print('\nNormalize   to    0-1 (float)')# very strange results if we dont
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test , axis=1)
#print('columns=', x_train.columns)
#print ('\ndf1=\n',x_train.loc[:, ['Open','High', 'Low', 'Close', 'range']])
#print ('\ndf1=\n',x_train.loc[:, ['sma10','sma20','sma50','sma200','range_sma']])
print(x_train[0])
print(x_train)
#print(x_train2[0])
#plot_image(x_test,'picture example')

print('\nRebalancing')




print('\nTransform data. Convert class vectors to binary class matrices (for ex. convert digit 7 to bit array[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]')
y_train = keras.utils.to_categorical(y_train, size_output)
y_test  = keras.utils.to_categorical(y_test, size_output)
print('y_train[0]=', y_train[0])
print('y_test [0]=',  y_test[0])

print('\ncreate model...')
print('\n======================================')
size_input   = x_train.shape[1] # no of features

model = Sequential()# stack of layers
#model.add(tf.keras.layers.Flatten())
model.add(Dense  (size_hidden, activation='relu', input_shape=(size_input,)))
model.add(Dropout(0.2))#for generalization
model.add(Dense  (size_hidden, activation='relu'))
model.add(Dropout(0.2))#regularization technic by removing some nodes
model.add(Dense  (size_output, activation='softmax'))# last layer always has softmax(except for regession problems and  binary- 2 classes where sigmoid is enough)
# For binary classification, softmax & sigmoid should give the same results, because softmax is a generalization of sigmoid for a larger number of classes.
# softmax:  loss: 0.3099 - acc: 0.8489 - val_loss: 0.2929 - val_acc: 0.8249
# sigmoid:  loss: 0.2999 - acc: 0.8482 - val_loss: 0.1671 - val_acc: 0.9863

# Prints a string summary of the  neural network.')
model.summary()
model.compile(loss      = 'categorical_crossentropy',# measure how accurate the model during training
              optimizer = RMSprop(),#this is how model is updated based on data and loss function
              metrics   = ['accuracy'])



layers = model.layers
#B_Output_Hidden = model.layers[0].get_weights()[1]
#print ('model.layers=\n', layers)
#print ('model.weights=\n', model.get_weights())

print('\ntrain model for ',epochs,' epochs...')
print('\n======================================')

history = model.fit(  x_train
                    , y_train
                    , batch_size     = batch_size
                    , epochs         = epochs
                    , validation_data= (x_test, y_test)
                    , verbose        = 1
                  # , callbacks=[early_stop, PrintDot()]#Early stopping is a useful technique to prevent overfitting.
                      )


print('\nEvaluate the model with unseen data. pls validate that test accuracy =~ train accuracy and close to 1.0')
print('\n======================================')
layers = model.layers
#B_Output_Hidden = model.layers[0].get_weights()[1]
#print ('model.layers=\n', layers)
print ('model.inputs=\n', size_input)
print ('model.output=\n', size_output)

print('\nplot_accuracy_loss_vs_time...')
history_dict = history.history
print(history_dict.keys())
plot_stat_loss_vs_time     (history_dict)
plot_stat_accuracy_vs_time (history_dict)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
plot_stat_train_vs_test(history)
plot_stat_loss_vs_accuracy(history_dict)

score = model.evaluate(x_test, y_test, verbose=0)                                     # random                                |  calc label
print('Test loss:    ', score[0], ' (is it close to 0?)')                            #Test,train loss     : 0.6938 , 0.6933   |  0.47  0.5
print('Test accuracy:', score[1], ' (is it close to 1 and close to train accuracy?)')#Test,train accuracy : 0.5000 , 0.5000   |  0.69, 0.74

print('\nPredict unseen data with 10 probabilities for 10 classes(choose the highest)')
Y_pred = model.predict(x_test)
print('labeled   as ', y_test[0], ' highest confidence for ' , np.argmax(y_test[0]))
print('predicted as ' ,Y_pred[0], ' highest confidence for ' , np.argmax(Y_pred[0]))
y_pred = np.argmax(Y_pred, axis=1)
Y_test = np.argmax(y_test, axis=1)
#print('prediction list= ' , y_pred.tolist())
#print('labelized  list= ' , Y_test.tolist())



filename='mlpt_'+symbol+'_'+str(epochs)+'_'+str(size_hidden)+'.model'
print('\nSave model as ',filename)
model.save(filename)# 5.4 mb





data = df_all['range'] #np.random.normal(0, 20, 1000)
bins = np.linspace(np.math.ceil(min(data)),    np.math.floor(max(data)),    100) # fixed number of bins

plot_histogram(  x = data
                  , bins=100
                  , title = 'range of a bar - Gaussian data (fixed number of bins)'
                  , xlabel ='range of a bar from open to close'
                  , ylabel ='count')


plot_histogram(    x = df_all['range_sma']
                    , bins=100
                    , title = 'diff bw 2 sma - Gaussian data (fixed number of bins)'
                    , xlabel ='diff bw 2 sma 10,20  '
                    , ylabel ='count')






np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')













