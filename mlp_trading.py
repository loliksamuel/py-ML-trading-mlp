from model.ml_model import MlModel
from model.ml_model_factory import MlModelFactory
from utils.utils import get_data_from_disc, plot_selected, plot_histogram, normalize0
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
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
    def execute(self, skip_days=3600,
                model_type=MlModel.MLP,
                epochs=5000,
                size_hidden=15,
                batch_size=128,
                percent_test_split=0.33,
                loss='categorical_crossentropy',
                lr=0.00001,  # default=0.001   best=0.00002
                rho=0.9,  # default=0.9     0.5 same
                epsilon=None,
                decay=0.0,
                kernel_init='glorot_uniform',
                dropout=0.2,
                verbose=0):
        print('\n======================================')
        print('\nLoading the data')
        print('\n======================================')
        df_all = self._data_load(skip_days)

        print('\n======================================')
        print('\nPlotting features')
        print('\n======================================')
        self._plot_features(df_all)

        print('\n======================================')
        print('\nSplitting the data to train & test data')
        print('\n======================================')
        self._data_split(df_all, percent_test_split)

        print('\n======================================')
        print('\nLabeling the data')
        print('\n======================================')
        self._data_label(df_all, percent_test_split)

        print('\n======================================')
        print('\nCleaning the data')
        print('\n======================================')
        self._data_clean()

        print('\n======================================')
        print('\nNormalizing the data')
        print('\n======================================')
        self._data_normalize()

        print('\n======================================')
        print('\nRebalancing Data')
        print('\n======================================')
        self._data_rebalance()

        print('\n======================================')
        print('\nTransform data. Convert class vectors to binary class matrices (for ex. convert digit 7 to bit array['
              '0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]')
        print('\n======================================')
        self._data_transform()

        print('\n======================================')
        print('\nCreating the model')
        print('\n======================================')
        model = MlModelFactory().create(model_type=model_type, size_hidden=size_hidden, size_input=self.size_input,
                                        size_output=self.size_output, dropout=dropout, kernel_init=kernel_init)

        print('\n======================================')
        print('\nCompiling the model')
        print('\n======================================')
        model.compile(loss=loss, lr=lr, rho=rho, epsilon=epsilon, decay=decay)

        print('\n======================================')
        print(f"\nTrain model for {epochs} epochs...")
        print('\n======================================')
        model.fit(x_train=self.x_train, y_train=self.y_train, x_test=self.x_test, y_test=self.y_test, epochs=epochs,
                  batch_size=batch_size, verbose=verbose)

        print('\n======================================')
        print('\nPrinting history')
        print('\n======================================')
        # model Loss, accuracy over time_hid003_RMS0.00001_epc5000_batch128_+1hid
        # model Loss, accuracy over time_hid003_RMS0.00001_epc5000_batch128_+1hid
        params = f'_hid{size_hidden}_RMS{lr}_epc{epochs}_batch{batch_size}_dropout{dropout}_sym{self.symbol}_inp{self.size_input}_out{self.size_output}_{model_type}'
        model.plot_evaluation(size_input=self.size_input, size_output=self.size_output, title=params)

        print('\n======================================')
        print('\nEvaluate the model with unseen data. pls validate that test accuracy =~ train accuracy and near 1.0')
        print('\n======================================')
        model.evaluate(x_test=self.x_test, y_test=self.y_test)

        print('\n======================================')
        print('\nPredict unseen data with 2 probabilities for 2 classes(choose the highest)')
        print('\n======================================')
        model.predict(x_train=self.x_train, x_test=self.x_test, y_test=self.y_test)

        print('\n======================================')
        print('\nSaving the model')
        print('\n======================================')
        model.save(folder='files/output/', filename=params)

        print('\n======================================')
        print('\nPlotting histograms')
        print('\n======================================')
        self._plot_features_hstg(df_all)

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _data_load(self, skip_days):
        df_all = get_data_from_disc(self.symbol, skip_days, size_output=self.size_output)
        # print('df_all.shape=',df_all.shape)
        # df_all = format_to_lstm(df_all)
        # print('df_all.shape=',df_all.shape)
        # columns = [df_all.shift(i) for i in range()]
        # df_all = pd.concat(columns, axis=1)
        # samples    = df_all.shape[0]
        # timestamps = 3
        # features   = df_all.shape[1]
        # #df_all = df_all.reshape ((df_all.shape[0], df_all.shape[1], 1))
        #  https://www.oipapio.com/question-3322022
        # df_all = df_all.values.reshape(samples,timestamps,features)
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
    def _data_split(self, df_all, percent_test_split):
        elements = df_all.size
        shape = df_all.shape

        df_data = df_all.loc[:, self.names_input]
        # today = datetime.date.today()
        # file_name = self.symbol+'_prepared_'+str(today)+'.csv'
        # df_data.to_csv(file_name)
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
    def _data_label(self, df_all, percent_test_split):
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
    def _data_clean(self):
        pass

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _data_normalize(self):
        self.x_train = normalize0(self.x_train, axis=1)
        self.x_test = normalize0(self.x_test, axis=1)
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
    def _data_rebalance(self):
        pass

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _data_transform(self):
        self.y_train = to_categorical(self.y_train, self.size_output)
        self.y_test = to_categorical(self.y_test, self.size_output)
        print('self.y_train[0]=', self.y_train[0])
        print('self.y_test [0]=', self.y_test[0])

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    @staticmethod
    def _plot_features_hstg(df_all):
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
