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
1        | why accuracy so high?? 64% and why so volatile ?
2        | should use PURGED K-FOLD Cross Validation or  TimeSeriesSplit instead of standard split
3        | which normalize function to use?
4        | grid search
5        | confusion matrix
6        | ensamble LSTM+RNN
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from buld.utils import *
from scipy import stats
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop

#import pandas.io.data as web

# https://towardsdatascience.com/deep-learning-for-beginners-practical-guide-with-python-and-keras-d295bfca4487
# https://www.youtube.com/watch?v=aircAruvnKk
#best so far: '^GSPC' 3600 5000 15  128 0.00001 0.9 None 0.0 glorot_uniform
symbol       = '^GSPC' #^GSPC=SP500 (3600->1970 or 12500 =2000) DJI(300, 1988)  QQQ(300, 2000) GOOG XLF XLV
skipDays     = 3600#12500 total 13894 daily bars
epochs       = 5000# 50 5000 #  iterations. on each, train all data, then evaluate, then adjust parameters (weights and biases)
size_hidden  = 15#below 10 it is underfit
# 15 20 512 # If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to overfit
batch_size   = 128# we cannot pass the entire data into network at once , so we divide it to batches . number of samples that we will pass through the network at 1 time and use for each epoch. default is 32
loss         = 'categorical_crossentropy' # a little better results than binary_crossentropy
lr           = 0.00001#default=0.001   best=0.00002
rho          = 0.9 # default=0.9   0.5 same
epsilon      = None
decay        = 0.0
kernel_init  = 'glorot_uniform'#  'glorot_uniform'(default),'normal', 'uniform'
#iterations  = 60000/128
names_input   = ['nvo', 'mom5', 'mom10', 'mom20', 'mom50', 'sma10', 'sma20', 'sma50', 'sma200', 'sma400', 'range_sma', 'log_sma20', 'log_sma50', 'log_sma200', 'log_sma400', 'bb_hi10', 'bb_lo10', 'bb_hi20', 'bb_lo20', 'bb_hi50', 'bb_lo50', 'bb_hi200', 'bb_lo200', 'rel_bol_hi10', 'rel_bol_lo10', 'rel_bol_hi20', 'rel_bol_lo20', 'rel_bol_hi50', 'rel_bol_lo50', 'rel_bol_hi200', 'rel_bol_lo200', 'rsi10', 'rsi20', 'rsi50', 'rsi5', 'stoc10', 'stoc20', 'stoc50', 'stoc200']
names_output  = ['Green bar', 'Red Bar']#, 'Hold Bar']#Green bar', 'Red Bar', 'Hold Bar'
size_input   = len(names_input) # 39#x_train.shape[1] # no of features
size_output  = len(names_output)#  2 # there are 3 classes (buy.sell. hold) or (green,red,hold)
#iterations  = 60000/128
percentTestSplit = 0.33#33% from data will go to test
seed = 7
np.random.seed(seed)

print('size_input=',size_input)
print('size_output=',size_output)
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
df_all = get_data_from_disc(symbol, skipDays, size_output = size_output)
print(df_all.tail())

# Slice and plot
#plot_selected(df_all, [  'Close', 'sma200'], shouldNormalize=True, symbol=symbol)

# Slice and plot
plot_selected(df_all,           title='TA-price of '+symbol+' vs time'              , columns=[ 'Close',  'sma200'],  shouldNormalize=False, symbol=symbol)

plot_selected(df_all.tail(500), title='TA-sma 1,10,20,50,200 of '+symbol+' vs time' , columns=[  'Close', 'sma10', 'sma20', 'sma50',  'sma200',  'sma400', 'bb_hi10', 'bb_lo10', 'bb_hi20', 'bb_lo20', 'bb_hi50',  'bb_lo200', 'bb_lo50', 'bb_hi200'],  shouldNormalize=False, symbol=symbol)

plot_selected(df_all.tail(500), title='TA-range sma,bband of '+symbol+' vs time'    , columns=[  'range_sma', 'log_sma20', 'log_sma50', 'log_sma200',  'log_sma400', 'rel_bol_hi10',  'rel_bol_hi20', 'rel_bol_hi200', 'rel_bol_hi50'],  shouldNormalize=False, symbol=symbol)
plot_selected(df_all.tail(500), title='TA-rsi,stoc of '+symbol+' vs time'           , columns=[  'rsi10', 'rsi20', 'rsi50', 'rsi5', 'stoc10', 'stoc20', 'stoc50', 'stoc200'],  shouldNormalize=False, symbol=symbol)

#plot_selected(df, ['Date','Close']                                    , start_date, end_date, shouldNormalize=False)
elements = df_all.size
shape=df_all.shape

print('\nsplit to train & test data')
print('\n======================================')
#df_data = df_all.loc[:,   [ 'Open', 'High', 'Low', 'Close', 'range', 'sma10', 'sma20', 'sma50',  'sma200',  'range_sma']]close
df_data = df_all.loc[:, names_input]
df_y    = df_all.loc[:, 'isUp'     ]#np.random.randint(0,2,size=(shape[0], ))
print('columns=', df_data.columns)
print('\nall data describe=\n',df_data.describe())
print('shape=',str(shape), " elements="+str(elements), ' rows=',str(shape[0]))
#(x_train, x_test)  = train_test_split(df_data.values, test_size=percentTestSplit, shuffle=True)#shuffle=False in timeseries
#(y_train, y_test)  = train_test_split(df_y.values   , test_size=percentTestSplit, shuffle=True)
import sklearn.model_selection as model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(df_data.values, df_y.values, train_size=0.66,test_size=0.33,shuffle=False, random_state=101)

# tscv = TimeSeriesSplit(n_splits=5)
print('\ntrain data', x_train.shape)
print(x_train[0])
print(x_train[1])
print(x_train[-1])
print(x_train[-2])

#plot_image(x_train, 'picture example')
#plot_images(x_train, y_train, 'picture examples')
print('\ntest data', x_test.shape)
print(x_test[0])#0=9308
print(x_test[1])
print(x_test[-2])
print(x_test[-1])

print('\nLabeling')
print('\n======================================')

#y = np.random.randint(0,2,size=(shape[0], ))
#print(y)
#df_y = pd.DataFrame()#, columns=list('is_up'))
#df_y['isUp'] = y
print(df_y)

#
# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
# y = np.array([100, 200, 300, 400, 500, 600])
# tscv = TimeSeriesSplit(n_splits=5)
# print(tscv)  # doctest: +NORMALIZE_WHITESPACE
# TimeSeriesSplit(max_train_size=None, n_splits=5)
# for train_index, test_index in tscv.split(X):
#      print("TRAIN:", train_index, "TEST:", test_index)
#      X_train, X_test = X[train_index], X[test_index]
#      y_train, y_test = y[train_index], y[test_index]
#      print(X_train,' , ', X_test)
#      print(X_train,' , ', y_train)
#      print(len(X_train),' , ', len(X_test))
#
'''
TRAIN: [0] TEST: [1]
TRAIN: [0 1] TEST: [2]
TRAIN: [0 1 2] TEST: [3]
TRAIN: [0 1 2 3] TEST: [4]
TRAIN: [0 1 2 3 4] TEST: [5]
'''


print(df_y.tail())
print('\nlabel describe\n',df_y.describe())
print('\nlabel describe train\n', stats.describe(y_train))
print('\nlabel describe train\n', stats.describe(y_test))

#(x_train, y_train)  = train_test_split(df.as_matrix(), test_size=0.33, shuffle=False)

print('\ntrain labels',y_train.shape)
print(y_train[0])
print(y_train[1])

print('\ntest labels', y_test.shape)
print(y_test[0])
print(y_test[1])

print(x_train.shape[0], 'train samples')
print( x_test.shape[0], 'test  samples')


print('\nClean data)')
print('\n======================================')
#dataset.isna().sum()
#dataset = dataset.dropna()

print('\nNormalize   to    0.0-1.0 ')# very strange results if we dont
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test , axis=1)

# print('\ndata  describe train\n', stats.describe(x_train))
# print('\ndata  describe train\n', stats.describe(x_test))
#print('columns=', x_train.columns)
#print ('\ndf1=\n',x_train.loc[:, ['Open','High', 'Low', 'Close', 'range']])
#print ('\ndf1=\n',x_train.loc[:, ['sma10','sma20','sma50','sma200','range_sma']])
print(x_train[0])

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
model.add(Dense  (size_hidden, activation='relu', input_shape=(size_input,), kernel_initializer=kernel_init))#, 'glorot_uniform','normal', 'uniform'
#model.add(Dense  (size_hidden, activation='relu', input_shape=(size_input,), kernel_regularizer=regularizers.l2(0.01)))#make worse (checked with size_hidden=20,256)

model.add(Dropout(0.2))#for generalization.

model.add(Dense  (size_hidden, activation='relu'))

model.add(Dropout(0.2))#for generalization.

model.add(Dense  (size_hidden, activation='relu'))
#model.add(Dense  (size_hidden, kernel_regularizer=regularizers.l2(0.01), activation='relu'))#make worse
model.add(Dropout(0.2))#regularization technic by removing some nodes
# Add fully connected layer with a ReLU activation function and L2 regularization
#model.add(Dense(units=15, input_dim=15     ,  kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))#make worse. 0.01 determines how much we penalize higher parameter values.
#model.add(Dense(units=16,                     kernel_regularizer=regularizers.l2(0.001),activation='relu'))

#model.add(Dropout(0.2))#regularization technic by removing some nodes
#model.add(Dense(units=16,                     kernel_regularizer=regularizers.l2(0.01), activation='relu'))#make worse.
#model.add(Dense(units=16,                     kernel_regularizer=regularizers.l2(0.001),activation='relu'))
#model.add(Dense(units=16,                     activation='relu'))#not improve
model.add(Dense  (size_output, activation='softmax'))# last layer always has softmax(except for regession problems and  binary- 2 classes where sigmoid is enough)
#model.add(Dense  (size_output, activation='sigmoid'))# worse results on hidden=20
# For binary classification, softmax & sigmoid should give the same results, because softmax is a generalization of sigmoid for a larger number of classes.
# softmax:  loss: 0.3099 - acc: 0.8489 - val_loss: 0.2929 - val_acc: 0.8249
# sigmoid:  loss: 0.2999 - acc: 0.8482 - val_loss: 0.1671 - val_acc: 0.9863

# Prints a string summary of the  neural network.')
model.summary()

#model.compile(loss      = 'log_loss',# no function
model.compile(loss      = loss,# a little better results than binaryEntropy
             # optimizer = SGD(lr=1, momentum=0.9),# make worse! try to increase your learning rate parameter etato force the network to converge faster to the optimum weights. Be aware though, if you increase it too much though, it will become unstable.
             # optimizer = SGD(lr=0.1, momentum=0.9),
             # optimizer = SGD(lr=0.01, momentum=0.9),
             # optimizer = SGD(lr=0.001, momentum=0.9),
             # optimizer = SGD(lr=0.0001, momentum=0.9),
             # optimizer = SGD(lr=0.00001, momentum=0.9),
             # Use dynamic learning rate

#model.compile(loss      = 'categorical_crossentropy',# measure how accurate the model during training
              # optimizer = RMSprop(lr=0.1),#this is how model is updated based on data and loss function
              # optimizer = RMSprop(lr=0.01),#this is how model is updated based on data and loss function
              # optimizer = RMSprop(lr=0.001),#this is default
              # optimizer = RMSprop(lr=0.0001),#interesting
                optimizer = RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay),#best
              #  optimizer = RMSprop(lr=0.00001,rho=0.5, epsilon=None, decay=0.0),#best default = rho=0-1, epsilon=1e-08, decay=0.0  #1e-06
              # optimizer = RMSprop(lr=0.00004),#interesting
              # optimizer = RMSprop(lr=0.000001),#bad
              #Adagrad(), Adadelta(), Adam(),Adamax(), Nadam(), TFOptimizer(),
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
                    , verbose        = 2
                  # , callbacks=[early_stop, PrintDot()]#Early stopping is a useful technique to prevent overfitting.
                      )


print('\nEvaluate the model with unseen data. pls validate that test accuracy =~ train accuracy and near 1.0')
print('\n======================================')
layers = model.layers
#B_Output_Hidden = model.layers[0].get_weights()[1]
#print ('model.layers=\n', layers)
print ('size.model.features(size_input) =\n', size_input)
print ('size.model.target  (size_output)=\n', size_output)

print('\nplot_accuracy_loss_vs_time...')
history_dict = history.history
print(history_dict.keys())
plot_stat_loss_vs_time     (history_dict)
plot_stat_accuracy_vs_time (history_dict)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plot_stat_loss_vs_accuracy(history_dict)
score = model.evaluate(x_test, y_test, verbose=1)                                     # random                                |  calc label
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

x_all = np.concatenate((x_train, x_test), axis=0)
Y_pred = model.predict(x_all)
print('labeled   as ', y_test[0], ' highest confidence for ' , np.argmax(y_test[0]))
print('predicted as ' ,Y_pred[0], ' highest confidence for ' , np.argmax(Y_pred[0]))
y_pred = np.argmax(Y_pred, axis=1).tolist()
Y_test = np.argmax(y_test, axis=1).tolist()

# print (type(y_pred))
# print (type(Y_test))
#
# # y1=y_pred.tolist()
# # y2=Y_test.tolist()
# y1=np.array(y_pred)
# y2=np.array(Y_test)
#
# yb = y1 ==  y2
# print('prediction list= ' , y1)
# print('labelized  list= ' , y2)
# print('result     list= ' , yb)
# print (type(y1))
# print (type(y2))
# print (type(yb))
# yb = np.asarray(yb)
# print (type(yb))
# plot_barchart2  (np.asarray(yb),  title="BT_pred vs observed", ylabel="x", xlabel="result")
# #
# n_folds = 10
# cv_scores, model_history = list(), list()
# for _ in range(n_folds):
#     # split data
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state = np.random.randint(1,1000, 1)[0])
#     # evaluate model
#     model, test_acc = evaluate_model(X_train, X_val, y_train, y_val)
#     print('>%.3f' % val_acc)
#     cv_scores.append(val_acc)
#     model_history.append(model)
#
# print('Estimated Accuracy %.3f (%.3f)' % (np.mean(cv_scores), np.std(cv_scores)))

dir = 'files/output/'
filename='mlpt_'+symbol+'_epc'+str(epochs)+'_hid'+str(size_hidden)+'_inp'+str(size_input)+'_out'+str(size_output)+'.model'
print('\nSave model as ',dir,'',filename)
model.save(dir+filename)# 5.4 mb





data = df_all['range'] #np.random.normal(0, 20, 1000)
bins = np.linspace(np.math.ceil(min(data)),    np.math.floor(max(data)),    100) # fixed number of bins

plot_histogram(  x = data
                  , bins=100
                  , title = 'TA-diff bw open and close - Gaussian data '
                  , xlabel ='range of a bar from open to close'
                  , ylabel ='count')


plot_histogram(    x = df_all['range_sma']
                    , bins=100
                    , title = 'TA-diff bw 2 sma - Gaussian data'
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













