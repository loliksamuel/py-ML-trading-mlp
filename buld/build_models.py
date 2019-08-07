import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.optimizers import RMSprop
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics.classification import _check_targets
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import eli5
from eli5.sklearn import PermutationImportance
from scipy import stats
from IPython.display import display
from mpl_toolkits.mplot3d import Axes3D
from buld.utils import data_load_and_transform, plot_selected, normalize1, plot_stat_loss_vs_accuracy, plot_conf_mtx, \
    plot_histogram, normalize2, normalize3, plot_stat_loss_vs_accuracy2, precision_threshold, recall_threshold, \
    plot_roc


class MlpTrading_old(object):
    def __init__(self) -> None:
        super().__init__()
        self.precision = 4
        np.set_printoptions(precision=self.precision)
        np.set_printoptions(suppress=True) #prevent numpy exponential #notation on print, default False
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.seed = 7
        np.random.seed(self.seed)

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def execute(self, symbol       = '^GSPC'
                    , modelType    = 'mlp'#mlp lstm drl xgb 'grid','scikit',
                    , names_input  = ['nvo']
                    , names_output = ['Green bar', 'Red Bar']# # , 'Hold Bar']#Green bar', 'Red Bar', 'Hold Bar'
                    , skip_days    = 3600
                    , epochs       = 5000
                    , size_hidden  = 15
                    , batch_size   = 128
                    , percent_test_split=0.33
                    , loss         = 'categorical_crossentropy'
                    , lr           = 0.00002# default=0.001   best=0.00002
                    , rho          = 0.9    # default=0.9     0.5 same
                    , epsilon      = None
                    , decay        = 0.0
                    , kernel_init  = 'glorot_uniform'
                    , dropout      = 0.2
                    , verbose      = 0

                    , activation='softmax'#sigmoid'
                    , use_random_label = False

                ):
        self.symbol = symbol

        self.names_output = names_output# , 'Hold Bar']#Green bar', 'Red Bar', 'Hold Bar'
        self.names_input  = names_input
        self.size_output = len(self.names_output)
        self.size_input  = len(self.names_input)

        df_all = self.data_prepare(percent_test_split, skip_days, use_random_label, modelType)
        params = f'hid{size_hidden}_rms{lr}_epo{epochs}_bat{batch_size}_dro{dropout}_sym{self.symbol}_inp{self.size_input}_out{self.size_output}_{modelType}'
        print(f'running with modelType {modelType}')
        if modelType=='grid':
            model = self.model_grid_search()
        elif modelType == 'scikit':
            model = self.model_create_scikit(epochs=epochs, batch_size=batch_size, size_hidden=size_hidden, dropout=dropout, activation=activation, optimizer='rmsprop', params=params)
        elif modelType == 'xgb':
            model = self.model_create_xgb(epochs, params)

        elif modelType == 'mlp':
            model = self.model_create_mlp(size_input=self.size_input, size_output=self.size_output ,activation=activation, optimizer=RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay), loss=loss, init=kernel_init, size_hidden=size_hidden, dropout=dropout, lr=lr, rho=rho, epsilon=epsilon, decay=decay)
            model = self.model_fitt(model, batch_size=batch_size, epochs=epochs, verbose=verbose, params=params)
            print('\n======================================')
            print('\nSaving the model')
            print('\n======================================')
            self.model_save(model, params)
        elif modelType == 'lstm':
            model = self._model_create_lstm(activation=activation, optimizer=RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay),  loss=loss, init=kernel_init, size_hidden=size_hidden, dropout=dropout, lr=lr, rho=rho, epsilon=epsilon, decay=decay)
            model = self.model_fitt(model, batch_size=batch_size, epochs=epochs, verbose=verbose, params=params)
            print('\n======================================')
            print('\nSaving the model')
            print('\n======================================')
            self.model_save(model, params)
        else:
            print('unsupported model. exiting')
            exit(0)




    def model_create_xgb(self, epochs,params):
        # https://www.datacamp.com/community/tutorials/xgboost-in-python
        #model = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10,  n_estimators=10)

        print(self.y_train.shape)
        model = xgb.XGBClassifier(max_depth=17, min_child_weight=1, learning_rate=0.1, n_estimators=epochs, silent=True,
                                  objective='binary:logistic', gamma=0, max_delta_step=0, subsample=1,
                                  colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=0,
                                  scale_pos_weight=1, seed=1, missing=None)

        eval_set = [(self.x_train, self.y_train), (self.x_test, self.y_test)]
        model.fit(self.x_train, self.y_train, eval_metric=["error", "logloss"], verbose=True, eval_set=eval_set)
        preds = model.predict(self.x_test)  # cv_results = xgb.cv  # xgb.plot_importance(xg_reg)
        score = model.score(self.x_test, self.y_test)
        print(f'error=#(wrong cases)/#(all cases)= {score} ')

        #make predictions for test data
        y_pred = model.predict(self.x_test)
        predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        # retrieve performance metrics
        results = model.evals_result()
        epochs = len(results['validation_0']['error'])
        x_axis = range(0, epochs)
        # plot log loss
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        plt.savefig(f'files/output/{params}_logloss.png')

        # plot classification error
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        plt.ylabel('Classification Error')
        plt.title('XGBoost Classification Error')
        plt.savefig(f'files/output/{params}_error.png')

        return model


    def model_grid_search(self):
        print("use_grid_search")
        activations = [ 'sigmoid']#, 'softplus', 'softsign', 'sigmoid',  'tanh', 'hard_sigmoid', 'linear', 'relu']#best 'softmax', 'softplus', 'softsign'
        inits       = ['glorot_uniform']#, 'zero', 'uniform', 'normal', 'lecun_uniform',  'glorot_uniform',  'he_uniform', 'he_normal']#all same except he_normal worse
        optimizers = ['RMSprop']#, 'SGD', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] # same for all
        losses =     ['categorical_crossentropy']#, 'categorical_crossentropy']#['mse', 'mae']  #better=binary_crossentropy
        epochs =     [300]#,500]  # , 100, 150] # default epochs=1,  better=100
        batch_sizes =[128]#],150,200]  # , 10, 20]   #  default = none best=32
        size_hiddens = [1]#], 200, 300, 400, 600]  # 5, 10, 20] best = 100 0.524993 Best score: 0.525712 using params {'batch_size': 128, 'dropout': 0.2, 'epochs': 100, 'loss': 'binary_crossentropy', 'size_hidden': 100}
        dropouts =     [0.2]#, 0.2, 0.3, 0.4]  # , 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        weights =      [1, 2]  # , 3, 4, 5]
        lrs     =      [0.02]#,0.03, 0.05, 0.07]#,0.001,0.0001,1,0.1,0.00001]#best 0.01 0.001 0.0001
        rhos    =      [0.01, 0.1, 0.2, 0.6]#all same
        grid_params = dict( activation  = activations,
                            init=inits,
                            # weight_constraint=weights,
                            #optimizer=optimizers,
                            epochs=epochs,
                            batch_size=batch_sizes,
                            loss= losses,
                            size_hidden = size_hiddens,
                            #dropout=dropouts,
                            lr= lrs
                            #rho = rhos

        )
        sk_params = {'size_input': self.size_input ,  'size_output':self.size_output}
        model = KerasClassifier(build_fn=self.model_create_mlp, **sk_params)
        #perm = PermutationImportance(model, random_state=1).fit(self.x_test, self.y_test)
        # weights = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names=self.names_input)
        # print('\nweights=',weights)

        grid = GridSearchCV(estimator=model, param_grid=grid_params, cv=2)
        X = np.concatenate((self.x_train, self.x_test), axis=0)
        Y = np.concatenate((self.y_train, self.y_test), axis=0)
        grid_result = grid.fit(self.x_train, self.y_train, verbose=2)
        # summarize results
        print("Best score: %f using params %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with params: %r" % (mean, stdev, param))
        return model


    def model_create_scikit(self, epochs=100, batch_size=128, size_hidden=15, dropout=0.2, optimizer='rmsprop', activation='sigmoid', params=''):
        sk_params = {'size_input': self.size_input, 'size_output': self.size_output, 'size_hidden': size_hidden, 'dropout': dropout,
                     'optimizer': optimizer, 'activation': activation}
        model = KerasClassifier(build_fn=self.model_create_mlp, **sk_params)
        history = model.fit(self.x_train, self.y_train, sample_weight=None, batch_size=batch_size, epochs=epochs,  verbose=1)  # validation_data=(self.x_test, self.y_test) kwargs=kwargs)
        self.model_weights(model, self.x_test, self.y_test, self.names_input)
        plot_stat_loss_vs_accuracy2(history.history)
        plt.savefig(f'files/output/{params}_Accuracy.png')
        score = model.score(self.x_test, self.y_test)
        print(f'accuracy= {score} ')
        self.model_predict(model, params=params)
        return model


    def model_fitt(self, model, batch_size=128, epochs=200, verbose=2, params=''):


        print('\n======================================')
        print(f"\nTrain model for {epochs} epochs...")
        print('\n======================================')
        history = self.model_fit(model, epochs, batch_size, verbose)


        print('\n======================================')
        print('\nPrinting history')
        print('\n======================================')
        # model Loss, accuracy over time_hid003_RMS0.00001_epc5000_batch128_+1hid
        # model Loss, accuracy over time_hid003_RMS0.00001_epc5000_batch128_+1hid
        #params = f'_hid{size_hidden}_RMS{lr}_epc{epochs}_batch{batch_size}_dropout{dropout}_sym{self.symbol}_inp{self.size_input}_out{self.size_output}_{modelType}'
        self.plot_evaluation( history, params=params)
        print('\n======================================')
        print('\nEvaluate the model with unseen data. pls validate that test accuracy =~ train accuracy and near 1.0')
        print('\n======================================')
        self.model_evaluate(model,self.x_test, self.y_test)
        print('\n======================================')
        print(f'\nPredict unseen data with {self.size_output} probabilities, for classes {self.names_output} (choose the highest)')
        print('\n======================================')
        self.model_predict(model,params=params)

        return model


    def data_prepare(self, percent_test_split, skip_days, use_random_label=False, modelType='mlp'):
        print('\n======================================')
        print('\nLoading the data')
        print('\n======================================')
        df_all = self._data_load(skip_days, self.size_output,  use_random_label)
        # print('\n======================================')
        # print('\nPlotting features')
        # print('\n======================================')
        # self._plot_features(df_all)
        print('\n======================================')
        print('\nSplitting the data to train & test data')
        print('\n======================================')
        self._data_split(df_all, percent_test_split)
        print('\n======================================')
        print('\nLabeling the data')
        print('\n======================================')
        self._label_split(df_all, percent_test_split)
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
        self._label_transform(modelType)
        return df_all




    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _data_load(self, skip_days, size_output, use_random_label=False):
        df_all = data_load_and_transform(self.symbol, usecols=['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], skip_first_lines = skip_days, size_output=size_output, use_random_label=use_random_label)
        # df_all = df_all.loc[:, self.names_input]
        # print('\ndf_all describe=\n', df_all.loc[:,
        #                               self.names_input].describe())

        # print('df_all.shape=',df_all.shape)
        # df_all = format_to_lstm(df_all)
        # print('df_all.shape=',df_all.shape)
        #columns = [df_all.shift(i) for i in range()]
        #df_all = pd.concat(columns, axis=1)
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

        plt.figure(figsize = (9, 5))
        df_all['rsi5'].plot(kind ="hist")
        plt.savefig('files/output/TA-rsi5.png')

        corrmat = df_all.corr()
        f, ax = plt.subplots(figsize =(9, 8))
        sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)

        #corrmat = df_all.corr()
        cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1);
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)
        plt.savefig('files/output/TA-corr.png')

        #np.plotScatterMatrix(df_all,10,20)

        # sns.set(style="ticks")
        # sns.pairplot(df_all)


# |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _data_split(self, df_all, percent_test_split):
        elements = df_all.size
        shape = df_all.shape

        df_data = df_all.loc[:, self.names_input]
        print (df_data.dtypes)
        df_data = df_data.round(self.precision)
        df_data.style.format("{:.4f}")
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
        print(self.x_test)

        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def _label_split(self, df_all, percent_test_split):
        df_y = df_all['isNextBarUp']  # np.random.randint(0,2,size=(shape[0], ))
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

        self.x_train = normalize3(self.x_train, axis=1)
        self.x_test  = normalize3(self.x_test , axis=1)
        # print('columns=', self.x_train.columns)
        # print ('\ndf1=\n',self.x_train.loc[:, ['Open','High', 'Low', 'Close', 'range0']])
        # print ('\ndf1=\n',self.x_train.loc[:, ['sma10','sma20','sma50','sma200','range_sma']])
        print('finished normalizing \n',stats.describe(self.x_train))
        print('\ndfn0=\n',self.x_train[0])
        print('\ndfn=\n',self.x_train)

        '''
finished normalizing  DescribeResult
(nobs=9315, 
            'nvo', 'mom5', 'mom10', 'mom20', 'mom50',  'range_sma', 'range_sma1', 'range_sma2', 'range_sma3', 'range_sma4', 'rel_bol_lo10', 'rel_bol_hi20', 'rel_bol_lo20', 'rel_bol_hi50', 'rel_bol_lo50'  'rel_bol_hi200',  'rel_bol_lo200', 'rsi10', 'rsi20', 'rsi50', 'rsi5', 'stoc10', 'stoc20', 'stoc50', 'stoc200'
min=       [ 0.63, -0.21,  -0.21,  -0.18   ,  -0.18  , -0.  , -0.  , -0.  , -0.  ,-0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  , -0.  ,  0.  ,0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]), 
max=       [ 1.  , -0.  , -0.  , -0.  , -0.          ,  0.  ,  0.  ,  0.  ,  0.  ,0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.27,0.26,  0.24,  0.26,  0.26,  0.29,  0.29,  0.3 ])), 
mean=      [ 0.99, -0.02, -0.02, -0.02, -0.02,         -0.  ,  0.  ,  0.  ,  0.  ,0.  ,  0.  , -0.  ,  0.  , -0.  ,  0.  , -0.  ,  0.  ,  0.02,0.02,  0.02,  0.02,  0.02,  0.02,  0.02,  0.03]), 
var =      [ 0.  , 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 0., 0.]), 
      
      normalize0*100
 min([ 0.6276, -0.2057, -0.209 , -0.1825, -0.1757, -0.004 , -0.005 ,-0.0077, -0.0164, -0.0114, -0.0025, -0.0147, -0.0027, -0.026 ,-0.0041, -0.0437, -0.0075,  0.0002,  0.0004,  0.0009, 0.0001,0.    ,  0.    ,  0.    ,  0.    ]), 
 max([ 1.    , -0.    , -0.    , -0.    , -0.    ,  0.0039,  0.005 ,    0.0096,  0.0188,  0.0214,  0.0097,  0.0013,  0.021 ,  0.0024, 0.0356,  0.012 ,  0.0597,  0.2654,  0.2613,  0.2395,  0.259 , 0.2555,  0.2868,  0.2934,  0.3028])), 
 mean([ 0.9934, -0.021 , -0.0204, -0.0196, -0.0186, -0.    ,  0.    ,0.0001,  0.0005,  0.0009,  0.0005, -0.0009,  0.0009, -0.0015,0.0017, -0.003 ,  0.0042,  0.0223,  0.0223,  0.0223,  0.0222,0.0225,  0.0233,  0.0243,  0.0268]), 
 var([0.0002, 0.0006, 0.0006, 0.0006, 0.0006, 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.0005, 0.0005, 0.0004, 0.0006, 0.0006, 0.0006, 0.0006, 0.0008]), 
       
       normalize3
 min([ -1.583 ,  -1.8729,  -1.9074,  -1.9675,  -2.1006, -20.2001, -11.8513,  -7.7558,  -4.4229,  -3.6384, -21.8447,  -5.0579,-16.8103,  -7.6295, -10.3472,  -7.6404,  -7.0404,  -3.5015, -2.4874,  -2.9664,  -3.5163,  -1.9537,  -1.9074,  -1.9675,-2.1006,  -2.2853]), 
 max([7.0308, 1.5706, 1.5023, 1.4322, 1.3643, 5.3087, 4.172 , 2.9205,2.0347, 1.8618, 2.0737, 8.4973, 1.7524, 6.2461, 1.5518, 4.7587,1.6522, 2.7927, 2.1145, 2.4315, 2.7735, 1.688 , 1.5023, 1.4322,1.3643, 1.1531])), 
 mean([-0.,  0.,  0.,  0., -0.,  0.,  0.,  0., -0.,  0., -0.,  0., -0.,-0.,  0.,  0., -0.,  0.,  0.,  0., -0., -0.,  0.,  0., -0., -0.]), 
 var([1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001,1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001])
 
        '''
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
    def _label_transform(self, modelType):
        print(f'categorizing   {self.size_output} classes')
        print('y_test=',self.y_test)
        # self.y_train = shift(self.y_train,1)
        # self.y_test  = shift(self.y_test,1)
        self.y_train_bak = self.y_train
        if modelType != 'xgb':
            self.y_train = to_categorical(self.y_train, num_classes=self.size_output)
            self.y_test  = to_categorical(self.y_test , num_classes=self.size_output)
        print(f'y_train[0]={self.y_train[0]}, it means label={np.argmax(self.y_train[0])}')
        print(f'y_test[0]={self.y_test[0]}, it means label={np.argmax(self.y_test[0])}')

    # |--------------------------------------------------------|
    # |                                            |
    # |--------------------------------------------------------|
    def model_create_mlp(self, size_input=28 ,  size_output=2 ,  activation='relu', optimizer='rmsprop', loss='categorical_crossentropy', init='glorot_uniform', size_hidden=15, dropout=0.2, lr=0.001, rho=0.9, epsilon=None, decay=0.0):

        model = Sequential()  # stack of layers

        model.add(Dense  (units=size_hidden, activation=activation, input_shape=(size_input,), kernel_initializer=init))
        model.add(Dropout(dropout))  # for generalization

        model.add(Dense  (units=size_hidden, activation=activation))
        model.add(Dropout(dropout))#for generalization.

        model.add(Dense  (units=size_hidden, activation=activation))
        model.add(Dropout(dropout))  # regularization technic by removing some nodes
        print(f'in={self.size_input} out={self.size_output} hid={size_hidden} lr={lr} rho={rho}, eps={epsilon}, dec={decay} activation={activation} optimizer={optimizer} loss={loss} init={init} ')
        model.add(Dense  (units=size_output, activation=activation))
        #model.summary()

        self._model_compile(model, optimizer=RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay),  loss=loss, lr=lr, rho=rho, epsilon=epsilon, decay=decay)

        return model



    def _model_create_lstm(self, activation='relu', optimizer='rmsprop', loss='categorical_crossentropy', init='glorot_uniform', size_hidden=50, dropout=0.2,  lr=0.001, rho=0.9, epsilon=None, decay=0.0):
        #input_shape = (input_length, input_dim)
        #input_shape=(self.size_input,)   equals to    input_dim = self.size_input
        #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        # X_train = []
        # Y_train = []
        #
        # for i in range(60, 2035):
        #     X_train.append(self.x_train[i-60:i, 0])
        #     Y_train.append(self.x_train[i, 0])
        # X_train, Y_train = np.array(X_train), np.array(Y_train)
        #
        # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        lookback = 2# Error when checking input: expected lstm_1_input to have shape (2, 39) but got array with shape (1, 39)
        lookback = 1

        print('b4 adding 1 dimension for lstm')
        # self.x_train = format_to_lstm(self.x_train, lookback)
        # self.x_test  = format_to_lstm(self.x_test, lookback)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], lookback, self.size_input))
        self.x_test  = np.reshape(self.x_test , (self.x_test.shape [0], lookback, self.size_input))

        #self.x_train, self.y_train = create_dataset(self.x_train, self.y_train, look_back = lookback)
        #self.x_test , self.y_test  = create_dataset(self.x_test , self.y_test , look_back = lookback)


        print(f'after adding {lookback} dimension for lstm')
        print('self.x_train.shape=',self.x_train.shape)#format_to_lstm(df)
        print('self.x_test.shape=',self.x_test.shape)#format_to_lstm(df)


        model = Sequential()

        model.add(LSTM(units = size_hidden, activation=activation, return_sequences = True, input_shape = (lookback, self.size_input)))
        model.add(Dropout(dropout))

        model.add(LSTM(units = size_hidden, activation=activation, return_sequences = True))
        model.add(Dropout(dropout))

        model.add(LSTM(units = size_hidden, activation=activation, return_sequences=False))
        model.add(Dropout(dropout))

        model.add(Dense  (units=self.size_output, activation=activation))
        #model.summary()

        self._model_compile(model, optimizer=optimizer,  loss=loss, lr=lr, rho=rho, epsilon=epsilon, decay=decay)

        return model


    # |--------------------------------------------------------|
    # |                                                  ,       |
    # |--------------------------------------------------------|
    @staticmethod
    def _model_compile(model,  optimizer='rmsprop', loss='categorical_crossentropy', lr=0.00001, rho=0.9, epsilon=None, decay=0.0):
        model.compile( loss=loss  # measure how accurate the model during training
                      ,optimizer=optimizer#RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)  # this is how model is updated based on data and loss function
                      # ,metrics=[precision_threshold(0.5)#, precision_threshold(0.2),precision_threshold(0.8)
                      #            , recall_threshold(0.5)#,    recall_threshold(0.2),   recall_threshold(0.8)
                      #           ]#https://stackoverflow.com/questions/42606207/keras-custom-decision-threshold-for-precision-and-recall
                       ,metrics=['accuracy']
                       )

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def model_fit(self, model, epochs=100, batch_size=128, verbose=0):
        return model.fit(self.x_train,
                         self.y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(self.x_test, self.y_test),
                         #validation_split = 0.1,
                         verbose=verbose)

    def model_weights(self, model, x, y, feature_names ):
        perm = PermutationImportance(model, random_state=1).fit(x, y)
        weights = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names=feature_names)
        print('\nweights=',weights)
    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def plot_evaluation(self, history, params=''):
        print(f'\nsize.model.features(size_input) = {self.size_input}')
        print(f'\nsize.model.target  (size_output)= {self.size_output}')
        print('\nplot_accuracy_loss_vs_time...')
        history_dict = history.history
        print(history_dict.keys())
        file_name=f'files/output/{params}_accuracy.png'
        plot_stat_loss_vs_accuracy(history_dict, title=f'{params}_accuracy')
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail())


    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def model_evaluate(self, model, x, y):
        score = model.evaluate(x,y,  verbose=0)
        print(f'Test loss:    {score[0]} (is it close to 0 ?)')
        print(f'Test accuracy:{score[1]} (is it close to 1 and close to train accuracy ?)')
        if self.size_output == 2:
           print(f'null accuracy={max(y.mean(), 1 - y.mean())}')# # calculate null accuracy (for multi-class classification problems)
#        elif self.size_output > 2:
 #           print(f'null accuracy={y.value_counts().head(1) / len(y)}')## calculate null accuracy (for multi-class classification problems)



    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def model_predict(self, model, params=''):
        y_pred_proba = model.predict      (self.x_test)# same as probs  = model.predict_proba(self.x_test)

        print(f'labeled   as {self.y_test[0]} highest confidence for index {np.argmax(self.y_test[0])}')
        print(f'predicted as {y_pred_proba[0]} highest confidence for index {np.argmax(y_pred_proba[0])}')
#        print('y_train class distribution',self.y_train.value_counts(normalize=True))
         #print(pd.DataFrame(y_pred).describe())
        x_all = np.concatenate((self.x_train, self.x_test), axis=0)
        y_pred_all = model.predict(x_all)
        print(f'labeled   as {self.y_test[0]} highest confidence for index {np.argmax(self.y_test[0])}')
        print(f'predicted as {y_pred_all[0]} highest confidence for index {np.argmax(y_pred_all[0])}')




        # Y_true = [0, 1, 0, 1]
        # Y_pred = [1, 1, 1, 0]
        # plot_conf_mtx(Y_true, Y_pred, self.names_output)
        y_pred_proba_r = y_pred_proba
        Y_true = np.argmax(self.y_test, axis=1)
        if isinstance(y_pred_proba[0],(np.int64)):#for scikit learn model
            Y_pred = y_pred_proba
        else:
            Y_pred = np.argmax(y_pred_proba, axis=1)
            print('proba=',y_pred_proba[:40])
            y_pred_proba_r = y_pred_proba[:, 1]
        plot_conf_mtx(Y_true, Y_pred, self.names_output, file_name=f'files/output/{params}_confusion.png')



        #print('probs=',y_pred_proba)
        if self.size_output == 2:#multiclass format is not supported
            plot_roc     (Y_true, Y_pred, y_pred_proba_r , file_name=f'files/output/{params}_roc.png')
        #y_pred_proba = 0.1 0.2 0.3
        #y_pred_proba_right = 0.1 0.2



        # print('Y_true=',Y_true[:40])
        # print('Y_pred=',Y_pred[:40])
        # print('accuracy=',accuracy_score(Y_true, Y_pred))
        #




        y_pred_proba_ok=[]
        for i, (p, t) in enumerate(zip(Y_pred,Y_true)):
            if p == t:
               y_pred_proba_ok.append(y_pred_proba_r[i])
        #       print(f'{i} ,{p} ,{t}, {y_pred_proba_r[i]} added' )
        #    else:
        #        print(f'{i} ,{p} ,{t}, {y_pred_proba_r[i]}')
        print (f'len {len(y_pred_proba_ok)} > {len(y_pred_proba_r)} = {len(Y_pred)} = {len(Y_true)}' )
        #print('y_pred_proba_ok=',y_pred_proba_ok[:40])#[0.59, 0.41], dtype=float32), array([0.45, 0.55], dtype=float32), array([0.3, 0.7], dtype=float32), array([0.42, 0.58]
        plot_histogram(y_pred_proba_r   , 20, f'{params}_predAll'   , 'Predicted probability', 'Frequency', xmin=0, xmax=1)
        plot_histogram(y_pred_proba_ok  , 20, f'{params}_predOk', 'Predicted probability', 'Frequency', xmin=0, xmax=1)







    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def model_save(self, model, filename):
        folder = 'files/output/'
        print(f'\nSave model as {folder}model{filename}.model')
        model.save(f'{folder}{filename}.model')

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    @staticmethod
    def _plot_features_hstg(df_all):
        plot_histogram(x=df_all['range0']
                       , bins=100
                       , title='TA-diff bw open and close - Gaussian data '
                       , xlabel='range of a bar from open to close'
                       , ylabel='count')

        plot_histogram(x=df_all['range_sma']
                       , bins=100
                       , title='TA-diff bw 2 sma - Gaussian data'
                       , xlabel='diff bw 2 sma 10,20  '
                       , ylabel='count')
