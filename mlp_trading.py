from tensorflow.python.keras.utils import normalize

from utils import get_data_from_disc, plot_selected, plot_stat_loss_vs_time, \
    plot_stat_accuracy_vs_time, plot_stat_loss_vs_accuracy, plot_histogram
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.optimizers import RMSprop
from ww import f
import pandas as pd
import numpy as np


class MlpTrading(object):
    def __init__(self, symbol) -> None:
        super().__init__()
        self.symbol = symbol

        self.names_input = ['nvo', 'mom5', 'mom10', 'mom20', 'mom50', 'sma10', 'sma20', 'sma50', 'sma200', 'sma400',
                            'range_sma', 'range_sma1', 'range_sma2', 'range_sma3', 'range_sma4', 'bb_hi10', 'bb_lo10',
                            'bb_hi20', 'bb_lo20', 'bb_hi50', 'bb_lo50', 'bb_hi200', 'bb_lo200', 'rel_bol_hi10',
                            'rel_bol_lo10', 'rel_bol_hi20', 'rel_bol_lo20', 'rel_bol_hi50', 'rel_bol_lo50',
                            'rel_bol_hi200',
                            'rel_bol_lo200', 'rsi10', 'rsi20', 'rsi50', 'rsi5', 'stoc10', 'stoc20', 'stoc50', 'stoc200']
        self.names_output = ['Green bar', 'Red Bar']  # , 'Hold Bar']#Green bar', 'Red Bar', 'Hold Bar'
        self.size_input = len(self.names_input)
        self.size_output = len(self.names_output)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.seed = 7
        np.random.seed(self.seed)

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def execute(self, skip_days    = 3600
                    , epochs       = 5000
                    , size_hidden  = 15
                    , batch_size   = 128
                    , percent_test_split=0.33
                    , loss         = 'categorical_crossentropy'
                    , lr           = 0.00001# default=0.001   best=0.00002
                    , rho          = 0.9    # default=0.9     0.5 same
                    , epsilon      = None
                    , decay        = 0.0
                    , kernel_init  = 'glorot_uniform'
                    , dropout      = 0.2
                    , verbose      = 0
    ):
        print('\n======================================')
        print('\nLoading the data')
        print('\n======================================')
        df_all = self._load_data(skip_days)

        print('\n======================================')
        print('\nPlotting features')
        print('\n======================================')
        self._plot_features(df_all)

        print('\n======================================')
        print('\nSplitting the data to train & test data')
        print('\n======================================')
        self._split_to_train_and_test_data(df_all, percent_test_split)

        print('\n======================================')
        print('\nLabeling the data')
        print('\n======================================')
        self._labeling(df_all, percent_test_split)

        print('\n======================================')
        print('\nCleaning the data')
        print('\n======================================')
        self._clean_data()

        print('\n======================================')
        print('\nNormalizing the data')
        print('\n======================================')
        self._normalize_data()

        print('\n======================================')
        print('\nRebalancing Data')
        print('\n======================================')
        self._rebalance_data()

        print('\n======================================')
        print('\nTransform data. Convert class vectors to binary class matrices (for ex. convert digit 7 to bit array['
              '0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]')
        print('\n======================================')
        self._transform_data()

        print('\n======================================')
        print('\nCreating the model')
        print('\n======================================')
        #model = self._create_model_mlp (size_hidden, dropout, kernel_init)
        model = self._create_model_lstm(size_hidden, dropout, kernel_init)

        print('\n======================================')
        print('\nCompiling the model')
        print('\n======================================')
        self._compile_mode(model, loss=loss, lr=lr, rho=rho, epsilon=epsilon, decay=decay)

        print('\n======================================')
        print(f"\nTrain model for {epochs} epochs...")
        print('\n======================================')
        history = self._train_model(model, epochs, batch_size, verbose)

        print('\n======================================')
        print('\nPrinting history')
        print('\n======================================')
        self._print_history(history)

        print('\n======================================')
        print('\nEvaluate the model with unseen data. pls validate that test accuracy =~ train accuracy and near 1.0')
        print('\n======================================')
        self._evaluate(model)

        print('\n======================================')
        print('\nPredict unseen data with 2 probabilities for 2 classes(choose the highest)')
        print('\n======================================')
        self._predict(model)

        print('\n======================================')
        print('\nSaving the model')
        print('\n======================================')
        self._save_model(model, epochs, size_hidden)

        print('\n======================================')
        print('\nPlotting histograms')
        print('\n======================================')
        self._plot_histograms(df_all)

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _load_data(self, skip_days):
        df_all = get_data_from_disc(self.symbol, skip_days, size_output=self.size_output)
        print(df_all.tail())
        return df_all

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _plot_features(self, df_all):
        plot_selected(df_all, title=f'TA-price of {self.symbol} vs time', columns=['Close', 'sma200'],
                      shouldNormalize=False, symbol=self.symbol)
        plot_selected(df_all.tail(500), title=f'TA-sma 1,10,20,50,200 of {self.symbol} vs time',
                      columns=['Close', 'sma10', 'sma20', 'sma50', 'sma200', 'sma400', 'bb_hi10', 'bb_lo10', 'bb_hi20',
                               'bb_lo20', 'bb_hi50', 'bb_lo200', 'bb_lo50', 'bb_hi200'], shouldNormalize=False,
                      symbol=self.symbol)
        plot_selected(df_all.tail(500), title=f'TA-range sma,bband of {self.symbol} vs time',
                      columns=['range_sma', 'range_sma1', 'range_sma2', 'range_sma3', 'range_sma4', 'rel_bol_hi10',
                               'rel_bol_hi20', 'rel_bol_hi200', 'rel_bol_hi50'], shouldNormalize=False,
                      symbol=self.symbol)
        plot_selected(df_all.tail(500), title=f'TA-rsi,stoc of {self.symbol} vs time',
                      columns=['rsi10', 'rsi20', 'rsi50', 'rsi5', 'stoc10', 'stoc20', 'stoc50', 'stoc200'],
                      shouldNormalize=False, symbol=self.symbol)

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _split_to_train_and_test_data(self, df_all, percent_test_split):
        elements = df_all.size
        shape = df_all.shape

        df_data = df_all.loc[:, self.names_input]
        print('columns=', df_data.columns)
        print('\ndata describe=\n', df_data.describe())
        print('shape=', str(shape), " elements=" + str(elements), ' rows=', str(shape[0]))
        (self.x_train, self.x_test) = train_test_split(df_data.values, test_size=percent_test_split,
                                                       shuffle=False)  # shuffle=False in timeseries
        # tscv = TimeSeriesSplit(n_splits=5)
        print('\ntrain data', self.x_train.shape)
        print(self.x_train[0])
        print(self.x_train[1])

        print('\ntest data', self.x_test.shape)
        print(self.x_test[0])
        print(self.x_test[1])

        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _labeling(self, df_all, percent_test_split):
        df_y = df_all['isUp']  # np.random.randint(0,2,size=(shape[0], ))
        print(df_y)
        (self.y_train, self.y_test) = train_test_split(df_y.values, test_size=percent_test_split, shuffle=False)

        print(df_y.tail())
        print('\nlabel describe\n', df_y.describe())
        # (self.x_train, self.y_train)  = train_test_split(df.as_matrix(), test_size=0.33, shuffle=False)

        print('\ntrain labels', self.y_train.shape)
        print(self.y_train[0])
        print(self.y_train[1])

        print('\ntest labels', self.y_test.shape)
        print(self.y_test[0])
        print(self.y_test[1])

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _clean_data(self):
        pass

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _normalize_data(self):
        self.x_train = normalize(self.x_train, axis=1)
        self.x_test = normalize(self.x_test, axis=1)
        # print('columns=', self.x_train.columns)
        # print ('\ndf1=\n',self.x_train.loc[:, ['Open','High', 'Low', 'Close', 'range']])
        # print ('\ndf1=\n',self.x_train.loc[:, ['sma10','sma20','sma50','sma200','range_sma']])
        print(self.x_train[0])
        print(self.x_train)
        # print(self.x_train2[0])
        # plot_image(self.x_test,'picture example')

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _rebalance_data(self):
        pass

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _transform_data(self):
        self.y_train = to_categorical(self.y_train, self.size_output)
        self.y_test = to_categorical(self.y_test, self.size_output)
        print('self.y_train[0]=', self.y_train[0])
        print('self.y_test [0]=', self.y_test[0])

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _create_model_mlp(self, size_hidden, dropout=0.2, kernel_init='glorot_uniform'):

        model = Sequential()  # stack of layers

        model.add(Dense  (units=size_hidden, activation='relu', input_shape=(self.size_input,), kernel_initializer=kernel_init))
        model.add(Dropout(dropout))  # for generalization

        model.add(Dense  (units=size_hidden, activation='relu'))
        model.add(Dropout(dropout))#for generalization.

        model.add(Dense  (units=size_hidden, activation='relu'))
        model.add(Dropout(dropout))  # regularization technic by removing some nodes

        model.add(Dense  (units=self.size_output, activation='softmax'))
        model.summary()
        return model

    def _create_model_lstm(self, size_hidden, dropout=0.2, kernel_init='glorot_uniform'):
#input_shape = (input_length, input_dim)
#input_shape=(self.size_input,)   equals to    input_dim = self.size_input
#X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
#X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        model = Sequential()

        model.add(LSTM(units = 50, activation='tanh', return_sequences = True, input_shape = (self.size_input, 1)))
        model.add(Dropout(dropout))

        model.add(LSTM(units = 50, activation='tanh', return_sequences = True))
        model.add(Dropout(dropout))

        model.add(LSTM(units = 50, activation='tanh', return_sequences=False))
        model.add(Dropout(dropout))

        model.add(Dense  (units=self.size_output, activation='softmax'))
        model.summary()
        return model

    def _create_model_lstm2(self, size_hidden, dropout=0.2, kernel_init='glorot_uniform'):
        #input_shape = (input_length, input_dim)
        #input_shape=(self.size_input,)   equals to    input_dim = self.size_input
        #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        model = Sequential()
        model.add(Embedding(input_dim = (self.size_input), output_dim = size_hidden, input_length = self.size_input))
        #model.add(Embedding( input_shape=(self.size_input,size_hidden), kernel_initializer=kernel_init))
        model.add(LSTM(units=size_hidden, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units=size_hidden, activation='sigmoid', inner_activation='hard_sigmoid'))
        model.add(Dropout(dropout))
        model.add(Dense(self.size_output, activation='softmax'))
        model.summary()
        return model
    # |--------------------------------------------------------|
    # |                                                  ,       |
    # |--------------------------------------------------------|
    @staticmethod
    def _compile_mode(model, loss='categorical_crossentropy', lr=0.00001, rho=0.9, epsilon=None, decay=0.0):
        model.compile( loss=loss  # measure how accurate the model during training
                      ,optimizer=RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)  # this is how model is updated based on data and loss function
                      ,metrics=['accuracy'])

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _train_model(self, model, epochs, batch_size, verbose=0):
        return model.fit(self.x_train,
                         self.y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(self.x_test, self.y_test),
                         #validation_split = 0.1,
                         verbose=verbose)

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _print_history(self, history):
        print(f'\nsize.model.features(size_input) = {self.size_input}')
        print(f'\nsize.model.target  (size_output)= {self.size_output}')

        print('\nplot_accuracy_loss_vs_time...')
        history_dict = history.history
        print(history_dict.keys())
        plot_stat_loss_vs_time(history_dict)
        plot_stat_accuracy_vs_time(history_dict)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail())

        plot_stat_loss_vs_accuracy(history_dict)

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _evaluate(self, model):
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f'Test loss:    {score[0]} (is it close to 0 ?)')
        print(f'Test accuracy:{score[1]} (is it close to 1 and close to train accuracy ?)')

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _predict(self, model):
        y_pred = model.predict(self.x_test)
        print(f'labeled   as {self.y_test[0]} highest confidence for {np.argmax(self.y_test[0])}')
        print(f'predicted as {y_pred[0]} highest confidence for {np.argmax(y_pred[0])}')

        x_all = np.concatenate((self.x_train, self.x_test), axis=0)
        y_pred = model.predict(x_all)
        print(f'labeled   as {self.y_test[0]} highest confidence for {np.argmax(self.y_test[0])}')
        print(f'predicted as {y_pred[0]} highest confidence for {np.argmax(y_pred[0])}')

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _save_model(self, model, epochs, size_hidden):
        folder = 'files/output/'
        filename = f'mlpt_{self.symbol}_epc{epochs}_hid{size_hidden}_inp{self.size_input}_out{self.size_output}.model'
        print(f'\nSave model as {folder}{filename}')
        model.save(f'{folder}{filename}')

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    @staticmethod
    def _plot_histograms(df_all):
        plot_histogram(x=df_all['range']
                       , bins=100
                       , title='TA-diff bw open and close - Gaussian data '
                       , xlabel='range of a bar from open to close'
                       , ylabel='count')

        plot_histogram(x=df_all['range_sma']
                       , bins=100
                       , title='TA-diff bw 2 sma - Gaussian data'
                       , xlabel='diff bw 2 sma 10,20  '
                       , ylabel='count')
