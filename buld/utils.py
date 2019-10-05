from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import pandas_datareader.data as pdr

import itertools
import os

from datetime import datetime
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns

import featuretools as ft
from featuretools import selection
from featuretools.primitives import MultiplyNumeric, Std, ModuloNumeric, Count
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from ta import *

from pycm import ConfusionMatrix
from scipy.special.cython_special import boxcox1p
from scipy.stats import boxcox_normmax

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support as scorex
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.utils import resample

from statsmodels.tsa.stattools import adfuller
from xgboost import plot_importance, XGBClassifier

from data.features import ProviderDateFormat
from data.features.features import Features


np.set_printoptions(suppress=True) #prevent numpy exponential #notation on print, default False
np.set_printoptions(precision=6)
np.set_printoptions(threshold=100)
# np.warnings.filterwarnings('ignore')
# np.seterr(divide='ignore', invalid='ignore')

#todo
def skew(df, features):
    for col in df.columns:
        df[col] = boxcox1p(df[col], boxcox_normmax(features[col] + 1)) if skew(df[col]) > 0.5 else df[col]

#todo
def fill(df):
    pass#fill with avg, median, most frequent

#
# def precision_threshold(threshold=0.5):
#     def precision(y_true, y_pred):
#         """Precision metric.
#         Computes the precision over the whole batch using threshold_value.
#         """
#         threshold_value = threshold
#         # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
#         y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
#         # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
#         true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
#         # count the predicted positives
#         predicted_positives = K.sum(y_pred)
#         # Get the precision ratio
#         precision_ratio = true_positives / (predicted_positives + K.epsilon())
#         return precision_ratio
#     return precision



#
# def recall_threshold(threshold = 0.5):
#     def recall(y_true, y_pred):
#         """Recall metric.
#         Computes the recall over the whole batch using threshold_value.
#         """
#         threshold_value = threshold
#         # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
#         y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
#         # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
#         true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
#         # Compute the number of positive targets.
#         possible_positives = K.sum(K.clip(y_true, 0, 1))
#         recall_ratio = true_positives / (possible_positives + K.epsilon())
#         return recall_ratio
#     return recall

def kpi_returns(prices)->float:
    return ((prices - prices.shift(-1)) / prices)[:-1]


def kpi_sharpeRatio()->float:
    risk_free_rate = 2.25  # 10 year US-treasury rate (annual) or 0
    sharpe = 2
    #  ((mean_daily_returns[stocks[0]] * 100 * 252) -  risk_free_rate ) / (std[stocks[0]] * 100 * np.sqrt(252))
    return sharpe


def kpi_commulativeReturn()->float:
    return 2.0


def kpi_risk(df)->float:
    return df.std()

def feature_selection(model, X_train, y_train, X_test, y_test):
    # Fit model using each importance as a threshold
    thresholds = pl.sort(model.feature_importances_)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


#https://machinelearningmastery.com/feature-importance-and-feature-selection-with-xgboost-in-python/
def plot_importance_xgb(xgb_model, title='feature importance xgb'):

    # plot feature importance
    plot_importance(xgb_model)
    plt.title(title)
    plt.savefig('files/output/' + title + '.png')

#This method of feature selection is applicable only when the input features are normalized and for linear svm https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
def plot_importance_svm(classifier, feature_names, top_features=None):
    if top_features == None:
        top_features = int(len(feature_names)/2)
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.clf()
    plt.figure(figsize=(25, 25))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(ticks=np.arange(1, 1 + 2 * top_features), labels=feature_names[top_coefficients], rotation=60, ha='right')
    title = f'top {top_features*2} features'
    plt.title(title)
    plt.savefig('files/output/' + title + '.png')


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red"#"white" if cm[i, j] > thresh else "black"
                 )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_histogram(x, bins, title, xlabel, ylabel, xmin=None, xmax=None):
    plt.clf()
    plt.hist(x, bins=bins)
    if xmin != None:
        plt.xlim(xmin, xmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('files/output/' + title + '.png')





def plot_roc(Y_true, Y_pred, probs, file_name='files/output/roc.png'):
    precision, recall, fscore, support = scorex(Y_true, Y_pred)
    auc                  = roc_auc_score(Y_true, probs)
    fpr, tpr, thresholds = roc_curve(Y_true, probs)
    print('auc       : %.3f' % auc)
    print('precision: {}'.format(precision))
    print('recall   : {}'.format(recall))
    print('fscore   : {}'.format(fscore))
    print('support  : {}'.format(support))
    plt.clf()
    plt.plot([0,1], [0,1], linestyle='--')
    plt.plot(fpr,tpr,'bo-', label = 'model');
    plt.plot(list(np.linspace(0, 1, num = 10)), list(np.linspace(0, 1, num = 10)), 'ro--', label = 'naive classifier');
    # for x, y, s in zip(fpr, tpr, thresholds):
    #     plt.text(x - 0.04,y + 0.02, s, fontdict={'size': 14});
    plt.legend(prop={'size':12})
    plt.ylabel('True Positive Rate', size = 12);
    plt.xlabel('False Positive Rate', size = 12);
    plt.title('AUC: %.3f' % auc, size = 12);#'Receiver Operating Characteristic Curve'
    plt.savefig(file_name)


def plot_conf_mtx(Y_true, Y_pred, target_names, file_name='files/output/Confusion matrix.png'):
    print("Regular/Normalized confusion matrix")
    count = len(Y_true)
    ones = np.count_nonzero(Y_true)
    zero = count - ones

    cm = confusion_matrix(Y_true, Y_pred).ravel()
    #tn, fp, fn, tp = cm.ravel()
    cm = ConfusionMatrix(actual_vector=Y_true, predict_vector=Y_pred)
    #cm.print_matrix()
    #cm.print_normalized_matrix()
    cnf_matrix = confusion_matrix(Y_true, Y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.clf()
    plt.figure()
    plt.subplot(1, 2, 1)
    title = 'not normalized'
    plot_confusion_matrix(cnf_matrix, classes=target_names,
                          title=title)
    plt.subplot(1, 2, 2)
    # plt.savefig('files/output/'+title+'.png')
    # Plot normalized confusion matrix
    # plt.figure()
    title = 'normalized'
    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,  title=title)

    plt.savefig(file_name)


def plot_barchart2(y, title="BT_pred vs observed", ylabel="Price", xlabel="Date"):
    l = len(y)
    greater_than_zero = y == True
    lesser_than_zero = y == False

    pl.clf()
    cax = pl.subplot(111)
    cax.bar(np.arange(l)[greater_than_zero], y[greater_than_zero], color='blue')
    cax.bar(np.arange(l)[lesser_than_zero], y[lesser_than_zero], color='red')
    pl.title(title + "TP+TN=" + str(sum(y)) + '#, ' + str(round(sum(y) / l * 100, 2)) + "%")
    pl.savefig('files/output/' + title + '.png')
    # pl.show()


def plot_selected(df, title='title', columns=[], shouldNormalize=True, symbol='any stock'):
    """Plot the desired columns over index values in the given range."""
    # df = df[columns][start_index:end_index]
    # df = df.loc[start_index:end_index, columns]
    df = df.loc[:, columns]
    ylabel = "Price"
    normal = "un normalized"
    if shouldNormalize:
        df = normalize(df.loc[:, ['Close', 'sma200']])
        ylabel = "%"
        normal = "normalized"
    # print('df.shape in plot=',df.shape)
    plot_data(df, title=title, ylabel=ylabel)


def plot_data(df, title="normalized Stock prices", ylabel="Price", xlabel="Date"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    plt.clf()
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig('files/output/' + title + '.png')


def plot_list(list, title="TA-normalized Stock prices", ylabel="Price", xlabel="Date", dosave=1):
    plt.plot(list)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if dosave == 1:
        plt.savefig('files/output/' + title + '.png')


def plot_barchart(list, title="BT", ylabel="Price", xlabel="Date", colors='green'):
    l = len(list)
    x = range(l)
    myarray = np.asarray(list)
    colors = colors  # 'green'#np.array([(1,0,0)]*l)
    # colors[myarray > 0.0] = (0,0,1)
    plt.bar(x, myarray, color=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('files/output/' + title + '.png')


def plot_image(df, title):
    plt.figure()
    plt.imshow(df[0])  # , cmap=plt.cm.binary)
    plt.colorbar()
    plt.gca().grid(False)
    plt.title(title)
    plt.show()


def plot_images(x, y, title):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=plt.cm.binary)
        plt.xlabel(y[i])
    plt.show()


def plot_stat_loss_vs_accuracy2(history_dict, title='model Loss, accuracy over time'):
    acc_train = history_dict['acc']
    #acc_test = history_dict['val_acc']
    loss_train = history_dict['loss']
    #loss_test = history_dict['val_loss']
    epochs = range(1, len(acc_train) + 1)

    plt.clf()
    plt.plot(epochs, loss_train, 'b', color='red', label='train loss')
    #plt.plot(epochs, loss_test, 'b', color='orange', label='test_loss')
    plt.plot(epochs, acc_train, 'b', color='green', label='train accuracy')
    #plt.plot(epochs, acc_test, 'b', color='blue', label='test  accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss & accuracy')
    plt.legend()



def plot_stat_loss_vs_accuracy(history_dict, title='model Loss, accuracy over time'):
    acc_train = history_dict['acc']
    acc_test = history_dict['val_acc']
    loss_train = history_dict['loss']
    loss_test = history_dict['val_loss']
    epochs = range(1, len(acc_train) + 1)

    plt.clf()
    plt.plot(epochs, loss_train, 'b', color='red'   , label='train loss')
    plt.plot(epochs, loss_test , 'b', color='orange', label='test_loss')
    plt.plot(epochs, acc_train , 'b', color='green' , label='train accuracy')
    plt.plot(epochs, acc_test  , 'b', color='blue'  , label='test  accuracy')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss & accuracy')
    plt.legend()
    plt.savefig('files/output/' + title + '.png')


def plot_stat_loss_vs_time(history_dict, title='model loss over time'):
    acc_train = history_dict['acc']
    acc_test = history_dict['val_acc']
    loss_train = history_dict['loss']
    loss_test = history_dict['val_loss']
    epochs = range(1, len(acc_train) + 1)

    plt.clf()
    plt.plot(epochs, loss_train, 'bo', color='red', label='train loss')
    # b is for "solid blue line"
    plt.plot(epochs, loss_test, 'b', color='red', label='test loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('files/output/' + title + '.png')


def plot_stat_accuracy_vs_time(history_dict, title='model accuracy over time'):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.clf()
    plt.plot(epochs, acc, 'bo', label='train acc')
    plt.plot(epochs, val_acc, 'b', label='test acc')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('files/output/' + title + '.png')


'''live plot of profits (a little bit slow)'''


def plot_live(cumsum, i):
    plt.plot(i, cumsum[i], '.b')  # - is line , b is blue
    plt.draw()
    plt.pause(0.01)


from tensorflow.python.keras.utils import normalize


# Test accuracy:0.68978194505275206
def data_normalize0(x, axis=1):
    print('\n============================================================================')
    print(f'#normalizing data axis = {axis}')
    print('===============================================================================')
    xn = normalize(x, axis=1)
    print ('xn=',xn)
    return xn


# normalize to first row  : Test accuracy:0.4978194505275206
def normalize1(df, axis)-> pd.DataFrame:
    return df / df.iloc[0, :]  # df/df[0]


def normalize2(df, axis)-> pd.DataFrame:
    train_stats = df.describe()
    return (df - train_stats['mean']) / train_stats['std']


def normalize3(x, axis):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)

    #x_norm = scaler.fit(x)
    #x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)
    return x_norm

def normalize_min_max2(df)-> pd.DataFrame:
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.transform(df)

def normalize_min_max(x):
    normalized = (x-min(x))/(max(x)-min(x))
    return normalized

def normalize_by_column(x, axis=0):
    min = np.min(x, axis=axis)
    max = np.max(x, axis=axis)
    return (x - min) / (max - min)

def symbol_to_path(symbol, base_dir="files/input"):
    """Return CSV file path given ticker symbol."""
    print('base_dir=', base_dir)
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data_from_disc_join(symbols, dates):#->pd.DaraFrame:
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


'from year 2000 only https://www.alphavantage.co'

def calc_scores(models, X, y):

    for model in models:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        print('\nmodel :',model)
        print("Accuracy Score: {0:0.2f} %".format(acc * 100))
        print("F1 Score: {0:0.4f}".format(f1))
        print("Area Under ROC Curve Score: {0:0.4f}".format(auc))


def calc_indicators2(symbol):
    YOUR_API_KEY = '7JRR5YWCLV4KGB9U'

    # Technical Indicators
    ti = TechIndicators(key='7JRR5YWCLV4KGB9U', output_format='pandas')
    ts = TimeSeries    (key='7JRR5YWCLV4KGB9U', output_format='pandas')
    sma, _ = ti.get_sma(symbol=symbol, interval='daily', time_period=20, series_type='close')
    wma, _ = ti.get_wma(symbol=symbol, interval='daily')
    ema, _ = ti.get_ema(symbol=symbol, interval='daily')
    macd, _ = ti.get_macd(symbol=symbol, interval='daily')
    stoc, _ = ti.get_stoch(symbol=symbol, interval='daily')
    rsi, _ = ti.get_rsi(symbol=symbol, interval='daily')
    adx, _ = ti.get_adx(symbol=symbol, interval='daily')
    cci, _ = ti.get_cci(symbol=symbol, interval='daily')
    aroon, _ = ti.get_aroon(symbol=symbol, interval='daily')
    bands, _ = ti.get_bbands(symbol=symbol, interval='daily')
    ad, _ = ti.get_ad(symbol=symbol, interval='daily')
    obv, _ = ti.get_obv(symbol=symbol, interval='daily')
    mom, _ = ti.get_mom(symbol=symbol, interval='daily')
    willr, _ = ti.get_willr(symbol=symbol, interval='daily')
    tech_ind = pd.concat([sma, ema, macd, stoc, rsi, adx, cci, aroon, bands, ad, obv, wma, mom, willr], axis=1)

    close = ts.get_daily(symbol=symbol, outputsize='full')[0]['close']  # compact/full
    direction = (close > close.shift()).astype(int)
    target = direction.shift(-1).fillna(0).astype(int)
    target.name = 'target'

    data = pd.concat([tech_ind, close, target], axis=1)

    return data


def calc_indicators(data, n):
    hh = data['high'].rolling(n).max()
    ll = data['low'].rolling(n).min()
    up, dw = data['close'].diff(), -data['close'].diff()
    up[up < 0], dw[dw < 0] = 0, 0
    macd = data['close'].ewm(12).mean() - data['close'].ewm(26).mean()
    macd_signal = macd.ewm(9).mean()
    tp = (data['high'] + data['low'] + data['close']) / 3
    tp_ma = tp.rolling(n).mean()
    indicators = pd.DataFrame(data=0, index=data.index,
                              columns=['sma', 'ema', 'momentum',
                                       'sto_k', 'sto_d', 'rsi',
                                       'macd', 'lw_r', 'a/d', 'cci'])
    indicators['sma'] = data['close'].rolling(10).mean()
    indicators['ema'] = data['close'].ewm(10).mean()
    indicators['momentum'] = data['close'] - data['close'].shift(n)
    indicators['sto_k'] = (data['close'] - ll) / (hh - ll) * 100
    indicators['sto_d'] = indicators['sto_k'].rolling(n).mean()
    indicators['rsi'] = 100 - 100 / (1 + up.rolling(14).mean() / dw.rolling(14).mean())
    indicators['macd'] = macd - macd_signal
    indicators['lw_r'] = (hh - data['close']) / (hh - ll) * 100
    indicators['a/d'] = (data['high'] - data['close'].shift()) / (data['high'] - data['low'])
    indicators['cci'] = (tp - tp_ma) / (0.015 * tp.rolling(n).apply(lambda x: np.std(x)))

    return indicators


def rebalance(unbalanced_data):
    # Separate majority and minority classes
    data_minority = unbalanced_data[unbalanced_data.target == 0]
    data_majority = unbalanced_data[unbalanced_data.target == 1]

    # Upsample minority class
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=5)

    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])

    data_upsampled.sort_index(inplace=True)

    # Display new class counts
    data_upsampled.target.value_counts()

    return data_upsampled

def plot_stats(self):
    print(f'df info=\n{self.df.info()}')
    sns.pairplot(data=self.df[['Close', 'Volume', 'CloseVIX', 'SKEW']])  # , hue="asset_price")
    for col in ['Close', 'Volume', 'CloseVIX', 'SKEW']:
        sns.distplot(self.df[col], bins=30)  # , hue="asset_price")
        #  plt.figure(i)
        sns.countplot(x=col, data=self.df)


def _sort_by_date(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    if inplace is True:
        formatted = df
    else:
        formatted = df.copy()

    formatted = formatted.sort_values('Date')#self.columns_map['Date'])

    return formatted


def print_is_stationary(df):
    # all columns seem to be stationary

    for col in df.columns:
        t = df[col].dtype
        if t != 'object' and t != 'str' :#and t != 'datetime' and t != 'string':
            adf     = adfuller(df[col], regression='ct')[0]
            p_value = adfuller(df[col], regression='ct')[1]
            is_stationary = p_value < 0.05
            print (f"adf={np.round(adf,1)}, p_value = {np.round(p_value,3)}. is_stationary={is_stationary} ( <0.05  means stationary) ) for column {col}. ")
        else:
            print(f'column {col} is not numeric. has type {t}.' )



def data_select(df, columns_input)->pd.DataFrame:
    print('\n============================================================================')
    print(f'#Selecting columns {columns_input}')
    print('===============================================================================')
    dfs = df[columns_input]
    print ('dfs=',dfs)
    return dfs

# def data_load_and_transform(symbol, usecols=['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], skip_first_lines = 1, size_output=2, use_random_label=False, feature_src= 'ta')->pd.DataFrame:
#     print('\n======================================')
#     print('\nLoading from disc raw data using ta(74 features) or talib(283 features)')
#     print('\n======================================')
#     df1 = get_data_from_disc(symbol, usecols)
#     dfc = data_clean(df1)
#     if feature_src == 'ta':
#         dft = data_transform(dfc, skip_first_lines ,size_output, use_random_label)
#     elif feature_src == 'talib':
#         fetures = Features()
#         dft  = fetures.add_features('all', dfc)
#     dft = create_target_label(dft, size_output, use_random_label)
#     print('\ndft describe=\n', dft.loc[:,  ['target' ]].describe())
#     #  https://www.oipapio.com/question-3322022
#     # df_all = df_all.values.reshape(samples,timestamps,features)
#     print(dft.tail())
#     return dft

def data_clean(df):
    # Clean NaN values
    print('\n============================================================================')
    print(f'#Cleaning NaN values')
    print('===============================================================================')
    dfc = utils.dropna(df)
    return dfc



def create_target_label(df1, size_output, use_random_label):
    df1 = df1.fillna(0)
    c0 = df1['Close']
    c1 = df1['Close'].shift(-1)
    df1['range0'    ] = c0  - c1
    df1['percentage'] = df1['range0'] / c1 * 100
    ## smart labeling
    if use_random_label == True:
        df1['isUp'] = np.random.randint(size_output, size=df1.shape[0])
    else:
        if size_output == 2:

            df1['isUp'] = (c0 > c1).astype(int)
            #df1['target'] = df1['isUp'].shift(-1).fillna(0).astype(int)



            #df1.loc[df1.range0  > 0.0, 'isUp'] = 1  # up
            #df1.loc[df1.range0 <= 0.0, 'isUp'] = 0  # dn
        elif size_output == 3:
            df1['isUp'] = 2  # hold
            df1.loc[df1.percentage >= +0.1, 'isUp'] = 1  # up
            df1.loc[df1.percentage <= -0.1, 'isUp'] = 0  # dn  # df1.loc[(-0.1 < df1.percentage <  +0.1), 'isUp'] =  0
        else:
            raise ValueError(f'Error. {size_output} is unsupported size_output. only 2 and 3 are supported')
    shift = -1  # -1
    df1['target'] = df1['isUp'].shift(shift)  # isNextBarUp: today's dataset  procuce  prediction is tommorow is up
    df1['target'] = df1['target'].fillna(0)  # .astype(int)#https://github.com/pylablanche/gcForest/issues/2
    df1['target'] = df1['target'].astype(int)

    #df1['isUp'  ] = df1['isUp'].astype(int)
    return df1


def get_data_from_disc(symbol, usecols=['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']):
    """Read stock data (adjusted close) for given symbols from CSV files.
    https://finance.yahoo.com/quote/%5EGSPC/history?period1=-630986400&period2=1563138000&interval=1d&filter=history&frequency=1d
    """
    print('\n\n\n============================================================================')
    print('#Loading raw data usecols=',usecols)
    print('===============================================================================')
    df1 = pd.read_csv(symbol_to_path(symbol)
                      , index_col='Date'
                      , parse_dates=True
                      , usecols=usecols
                      , na_values=['nan'])

    print('raw columns=', df1.columns)
    print('\nraw df1=\n', df1)
    return df1


# def get_data_from_web(symbol):
#     start, end = '1970-01-03','2019-07-12'#'2007-05-02', '2016-04-11'
#     data   = web.DataReader(symbol, 'yahoo', start, end)
#     data   = pd.DataFrame(data)
#     prices = data['Adj Close']
#     prices = prices.astype(float)
#     return prices
# def get_data_from_web2(symbol):
#     start, end = '1970-01-03', '2019-07-12'  # '2007-05-02', '2016-04-11'
#     data = pdr.get_data_yahoo(symbol, start, end)
#     closePrice = data["Close"]
#     print(closePrice)
#     return closePrice


def get_state(parameters, t, window_size=20):
    outside = []
    d = t - window_size + 1
    for parameter in parameters:
        block = (
            parameter[d: t + 1]
            if d >= 0
            else -d * [parameter[0]] + parameter[0: t + 1]
        )
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        for i in range(1, window_size, 1):
            res.append(block[i] - block[0])
        outside.append(res)
    return np.array(outside).reshape((1, -1))


# reshape  because   LSTM receives  [samples, look_back, features]
def format_to_lstm(df, look_back=1):
    X = np.array(df)
    return np.reshape(X
                      , (X.shape[0], look_back, X.shape[1]))


def format_to_lstm_regression(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):  # for i in range(look_back, len(dataset)):
        x = dataset[i:(i + look_back), 0]  # [i-look_back:i, 0]
        y = dataset[i + look_back, 0]  # [i, 0]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

def feature_tool(df_x):
    '''
    :param df_x: df
    :return: 80,089 features: 283 + ( 283/2*283 ) * 2
    #https://danwertheimer.github.io/rapid-model-prototyping-with-deep-feature-synthesis-and-xgboost
    '''
    print(f'start featuretools')

    # Make an entityset and add the entity
    es = ft.EntitySet(id = 'sp500')
    es = es.entity_from_dataframe( entity_id  = 'sp500'
                                 , dataframe  = df_x
                                 , make_index = True
                                 , index      = 'index'
                                 )

    # es.normalize_entity(base_entity_id='sp500',
    #                      new_entity_id='sessions',
    #                      index        ='session'
    #                       )


    primitives_aggregate      = [Std, Count]#'std', 'min', 'count', 'max', 'mean', 'median',  'mode', 'num_true', 'num_unique', 'sum','skew', 'percent_true', 'last', 'trend', 'n_most_common', 'time_since_last','avg_time_between'] #create a single value
    primitives_where          = ['std', 'min', 'max', 'mean', 'count']
    primitives_groupby        = ['cum_sum', 'cum_count', 'cum_mean', 'cum_min', 'cum_max'] #group by id  # [1, 2, 3, 4, 5]).tolist() = [1, 3, 6, 10, 15]
    primitives_transform      = [#'add_numeric'       #Element-wise       addition of 2 lists. create 283/2*283 = 40,044 new features
                                    MultiplyNumeric
                                , ModuloNumeric
                               #, 'multiply_numeric'  #Element-wise multiplication of 2 lists. create 283/2*283 = 40,044 new features
                               # 'subtract_numeric'  #Element-wise subtraction    of 2 lists. create 283/2*283 = 40,044 new features
                               # , 'modulo_numeric'    #Element-wise modulo         of 2 lists. create 283/2*283 = 40,044 new features
                               #, 'and'               #Element-wise logical AND    of 2 lists. create 283/2*283 = 40,044 new features
                               #, 'or'                #Element-wise logical OR     of 2 lists. create 283/2*283 = 40,044 new features
                               , 'absolute', 'percentile'#, 'cum_count', 'cum_sum', 'cum_mean', 'cum_min', 'cum_max', 'cum_mean'
                                 ]
    # 'absolute','percentile', 'cum_count', 'cum_sum', 'cum_mean', 'cum_min', 'cum_max', 'cum_mean', 'subtract', 'divide','time_since_previous', 'latitude', 'longitude', isin is_null is_weekend year week log]
    # Run deep feature synthesis with transformation primitives
    feature_matrix, feature_defs = ft.dfs(  entityset                 = es
                                          , target_entity             = 'sp500'

                                          , agg_primitives            = primitives_aggregate
                                          , trans_primitives          = primitives_transform
                                          , groupby_trans_primitives  = primitives_groupby
                                          , where_primitives          = primitives_where
                                          , max_features              = 89000
                                          #, drop_contains             = 'target'
                                          #, seed_features             = ['sepal length']
                                          , max_depth                 = 1
                                          , n_jobs                    = 1 #-1 will use all cores
                                          , verbose                   = True  )

    print(f'finished featuretools. feature_matrix=\n{feature_matrix.head()}')
    #print(f'finished2 es={es}')
    #print(f'finished3 es={es["sp500"]}')
    #print(f'feature_matrix.columns.tolist()={feature_matrix.columns.tolist()}')
    #print(f'ft.list_primitives() {ft.list_primitives()}')
    #print(f'ft.list_primitives() {ft.show_info()}')

    feature_matrix = selection.remove_low_information_features(feature_matrix)
    return feature_matrix

def reduce_mem_usage2(df):
    for col in df.columns:
        col_type = df[col].dtype
        col_items = df[col].count()
        col_unique_itmes = df[col].nunique()
        if (col_type == 'object') and (col_unique_itmes < col_items):
            df[col] = df[col].astype('category')
        if (col_type == 'int64'):
            df[col] = df[col].astype('int32')
        if (col_type == 'float64'):
            df[col] = df[col].astype('float32')
    return df


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def format_col_date(df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
    if inplace is True:
        formatted = df
    else:
        formatted = df.copy()

    COL_DATE = 'Date'#self.columns_map['Date']
    dfc = formatted.loc[:, COL_DATE]
    PP = ProviderDateFormat.ProviderDateFormat
    date_format = PP.DATETIME_HOUR_24
    if date_format is PP.TIMESTAMP_UTC:
        formatted[COL_DATE] = dfc.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M'))
        formatted[COL_DATE] = pd.to_datetime(dfc, format='%Y-%m-%d %H:%M')
    elif date_format is PP.TIMESTAMP_MS:
        formatted[COL_DATE] = pd.to_datetime(dfc, unit='ms')
    elif date_format is PP.DATETIME_HOUR_12:
        formatted[COL_DATE] = pd.to_datetime(dfc, format='%Y-%m-%d %I-%p')
    elif date_format is PP.DATETIME_HOUR_24:
        formatted[COL_DATE] = pd.to_datetime(dfc, format='%Y-%m-%d %H')
    elif date_format is PP.DATETIME_MINUTE_12:
        formatted[COL_DATE] = pd.to_datetime(dfc, format='%Y-%m-%d %I:%M-%p')
    elif date_format is PP.DATETIME_MINUTE_24:
        formatted[COL_DATE] = pd.to_datetime(dfc, format='%Y-%m-%d %H:%M')
    elif date_format is PP.DATE:
        formatted[COL_DATE] = pd.to_datetime(dfc, format='%Y-%m-%d')
    elif date_format is PP.CUSTOM_DATIME:
        formatted[COL_DATE] = pd.to_datetime(dfc, format=None, infer_datetime_format=True)
    else:
        raise NotImplementedError


    formatted[COL_DATE] = formatted[COL_DATE].values.astype(np.int64) // 10 ** 9

    return formatted
#
# y_pred = [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1,
#           0, 1, 0, 0, 1]
# Y_test = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0,
#           1, 0, 0, 1, 1]
# print(type(y_pred))
# print(type(Y_test))
# y1 = np.array(y_pred)
# y2 = np.array(Y_test)
# print(type(y_pred))
# print(type(Y_test))
# yb = np.array(y_pred) == np.array(Y_test)
# print(yb)
# print(type(yb))
# #plot_barchart2(yb, title="BT_pred vs observed", ylabel="x", xlabel="result")
