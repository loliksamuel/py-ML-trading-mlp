from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import os

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from pycm import ConfusionMatrix
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as scorex
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from ta import *

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True) #prevent numpy exponential #notation on print, default False

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

def kpi_returns(prices):
    return ((prices - prices.shift(-1)) / prices)[:-1]


def kpi_sharpeRatio():
    risk_free_rate = 2.25  # 10 year US-treasury rate (annual) or 0
    sharpe = 2
    #  ((mean_daily_returns[stocks[0]] * 100 * 252) -  risk_free_rate ) / (std[stocks[0]] * 100 * np.sqrt(252))
    return sharpe


def kpi_commulativeReturn():
    return 2


def kpi_risk(df):
    return df.std()


def plot_histogram(x, bins, title, xlabel, ylabel):
    plt.clf()
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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


def plot_hist_proba(y_pred_prob, file_name='files/output/probability histogram.png'):
    plt.clf()
    plt.hist(y_pred_prob, bins=20)

    # x-axis limit from 0 to 1
    plt.xlim(0,1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of diabetes')
    plt.ylabel('Frequency')
    plt.savefig(file_name)


def plot_roc(Y_true, Y_pred, probs, file_name='files/output/roc.png'):
    precision, recall, fscore, support = scorex(Y_true, Y_pred)
    auc = roc_auc_score(Y_true, probs)
    print('AUC: %.3f' % auc)
    fpr, tpr, thresholds = roc_curve(Y_true, probs)
    print('\nprecision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
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
    print('\nplot_conf_mtx')
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
    plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                          title=title)

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
def normalize1(df, axis):
    return df / df.iloc[0, :]  # df/df[0]


def normalize2(df, axis):
    train_stats = df.describe()
    return (df - train_stats['mean']) / train_stats['std']


def normalize3(x, axis):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)
    #x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)
    return x_norm


def symbol_to_path(symbol, base_dir="files/input"):
    """Return CSV file path given ticker symbol."""
    print('base_dir=', base_dir)
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data_from_disc_join(symbols, dates):
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


def calc_indicators2(symbol):
    YOUR_API_KEY = '7JRR5YWCLV4KGB9U'
    # Technical Indicators
    ti = TechIndicators(key='7JRR5YWCLV4KGB9U', output_format='pandas')
    ts = TimeSeries(key='7JRR5YWCLV4KGB9U', output_format='pandas')
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

def data_select(df, columns_input):
    print('\n============================================================================')
    print(f'#Selecting columns {columns_input}')
    print('===============================================================================')
    dfs = df[columns_input]
    print ('dfs=',dfs)
    return dfs

def data_load_and_transform(symbol, usecols=['Date', 'Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'], skip_first_lines = 1, size_output=2, use_random_label=False):
    df1 = get_data_from_disc(symbol, usecols)
    dfc = data_clean(df1)
    dft = data_transform(dfc, skip_first_lines ,size_output, use_random_label)
    return dft

def data_clean(df):
    # Clean NaN values
    print('\n============================================================================')
    print(f'#Cleaning NaN values')
    print('===============================================================================')
    dfc = utils.dropna(df)
    return dfc

def data_transform(df1, skip_first_lines = 400, size_output=2, use_random_label=False):
    if  skip_first_lines < 400:
        print (f'error: skip_first_lines must be > 400 existing...')
        exit(1)
    print('\n============================================================================')
    print(f'#Transform raw data skip_first_lines={skip_first_lines}, size_output={size_output}, use_random_label={use_random_label}')
    print('===============================================================================')


    # Add ta features filling NaN values
    # df1 = add_all_ta_features(df, "Open", "High", "Low", "Close",  fillna=True)#, "Volume_BTC",

    # Add bollinger band high indicator filling NaN values
    df1['bb_hi10'] = bollinger_hband_indicator(df1["Close"], n=10, ndev=2, fillna=True)
    df1['bb_lo10'] = bollinger_lband_indicator(df1["Close"], n=10, ndev=2, fillna=True)
    df1['bb_hi20'] = bollinger_hband_indicator(df1["Close"], n=20, ndev=2, fillna=True)
    df1['bb_lo20'] = bollinger_lband_indicator(df1["Close"], n=20, ndev=2, fillna=True)
    df1['bb_hi50'] = bollinger_hband_indicator(df1["Close"], n=50, ndev=2, fillna=True)
    df1['bb_lo50'] = bollinger_lband_indicator(df1["Close"], n=50, ndev=2, fillna=True)
    df1['bb_hi200'] = bollinger_hband_indicator(df1["Close"], n=200, ndev=2, fillna=True)
    df1['bb_lo200'] = bollinger_lband_indicator(df1["Close"], n=200, ndev=2, fillna=True)

    df1['rsi5'] = rsi(df1["Close"], n=5, fillna=True)
    df1['rsi10'] = rsi(df1["Close"], n=10, fillna=True)
    df1['rsi20'] = rsi(df1["Close"], n=20, fillna=True)
    df1['rsi50'] = rsi(df1["Close"], n=50, fillna=True)

    df1['stoc10'] = stoch(df1["High"], df1["Low"], df1["Close"], n=10, fillna=True)
    df1['stoc20'] = stoch(df1["High"], df1["Low"], df1["Close"], n=20, fillna=True)
    df1['stoc50'] = stoch(df1["High"], df1["Low"], df1["Close"], n=50, fillna=True)
    df1['stoc200'] = stoch(df1["High"], df1["Low"], df1["Close"], n=200, fillna=True)

    df1['mom5'] = wr(df1["High"], df1["Low"], df1["Close"], lbp=5, fillna=True)
    df1['mom10'] = wr(df1["High"], df1["Low"], df1["Close"], lbp=10, fillna=True)
    df1['mom20'] = wr(df1["High"], df1["Low"], df1["Close"], lbp=20, fillna=True)
    df1['mom50'] = wr(df1["High"], df1["Low"], df1["Close"], lbp=50, fillna=True)

    df1['sma10'] = df1['Close'].rolling(window=10).mean()  # .shift(1, axis = 0)
    df1['sma20'] = df1['Close'].rolling(window=20).mean()
    df1['sma50'] = df1['Close'].rolling(window=50).mean()
    df1['sma200'] = df1['Close'].rolling(window=200).mean()
    df1['sma400'] = df1['Close'].rolling(window=400).mean()
    # df1['mom']=pandas.stats.
    df1 = df1[-(df1.shape[0] - skip_first_lines):]  # skip 1st x rows, x years due to NAN in sma, range
    df1['nvo'] = df1['Volume'] / df1['sma10'] / 100  # normalized volume

    # df/df.iloc[0,:]
    df1['range_sma'] = (df1['Close'] - df1['sma10']) / df1['Close']*100
    df1['range_sma1'] = (df1['sma10'] - df1['sma20']) / df1[ 'sma10'] *100 # small sma above big sma indicates that price is going up
    df1['range_sma2'] = (df1['sma20'] - df1['sma50']) / df1['sma20'] *100 # small sma above big sma indicates that price is going up
    df1['range_sma3'] = (df1['sma50'] - df1['sma200']) / df1['sma50'] *100 # small sma above big sma indicates that price is going up
    df1['range_sma4'] = (df1['sma200'] - df1['sma400']) / df1['sma200'] *100 # small sma above big sma indicates that price is going up

    df1['rel_bol_hi10'] = (df1['High'] - df1['bb_hi10']) / df1['High']*100
    df1['rel_bol_lo10'] = (df1['Low'] - df1['bb_lo10']) / df1['Low']*100
    df1['rel_bol_hi20'] = (df1['High'] - df1['bb_hi20']) / df1['High']*100
    df1['rel_bol_lo20'] = (df1['Low'] - df1['bb_lo20']) / df1['Low']*100
    df1['rel_bol_hi50'] = (df1['High'] - df1['bb_hi50']) / df1['High']*100
    df1['rel_bol_lo50'] = (df1['Low'] - df1['bb_lo50']) / df1['Low']*100
    df1['rel_bol_hi200'] = (df1['High'] - df1['bb_hi200']) / df1['High']*100
    df1['rel_bol_lo200'] = (df1['Low'] - df1['bb_lo200']) / df1['Low']*100

    # df1['isUp'] = 0
    print(df1)
    # print ('\ndf1=\n',df1.tail())
    # print ('\nsma_10=\n',df1['sma10'] )
    # print ('\nsma_200=\n',df1['sma200'] )
    # print ('\nrsi10=\n',df1['rsi10'] )
    # print ('\nrsi5=\n',df1['rsi5'] )
    # print ('\nstoc10=\n',df1['stoc10'] )
    # print ('\nstoc200=\n',df1['stoc200'] )
    # print ('\nrangesma=\n',df1['rangesma'])
    # print ('\nrangesma4=\n',df1['rangesma4'])
    # print ('\nrel_bol_hi10=\n',df1['rel_bol_hi10'])
    # print ('\nrel_bol_hi200=\n',df1['rel_bol_hi200'])

    # df1['sma4002' ] = sma
    # df1['ema' ] = ema
    # df1['macd' ] = macd
    # df1['stoc' ] = stoc
    # df1['rsi' ] = rsi
    # tech_ind = pd.concat([sma, ema, macd, stoc, rsi, adx, cci, aroon, bands, ad, obv, wma, mom, willr], axis=1)

    ## labeling
    df1['range2'    ] = df1['Close'].shift(2)  - df1['Open'].shift(2) #df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']
    df1['range1'    ] = df1['Close'].shift(1)  - df1['Open'].shift(1) #df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']
    df1['range0'    ] = df1['Close'].shift(0)   - df1['Open'].shift(0) #df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']
    df1.loc[df1.range1  > 0.0, 'isPrev1Up'] = 1
    df1.loc[df1.range1 <= 0.0, 'isPrev1Up'] = 0
    df1.loc[df1.range2  > 0.0, 'isPrev2Up'] = 1
    df1.loc[df1.range2 <= 0.0, 'isPrev2Up'] = 0
    #df1['rangebug1'] = df1['Close'].shift(1)  - df1['Open'].shift(1) #bug!!! df1['Close'].shift() - df1['Open'].shift()  or df1['Close'].shift(1) - df1['Close']
    #df1['rangebug2'] = df1['Close'].shift(0)  - df1['Open'].shift(0) #bug!!!  need to use  df.loc[i-1, 'Close'] or df1['Close'] - df1['Close'].shift(1)
    df1 = df1.fillna(0)#https://github.com/pylablanche/gcForest/issues/2
    df1['percentage'] = df1['range0'] / df1['Open'] * 100
    ## smart labeling
    if use_random_label==True:
        df1['isUp']  = np.random.randint(size_output, size=df1.shape[0])
    else:
        if size_output == 2 :
            df1.loc[df1.range0  > 0.0, 'isUp'] = 1#up
            df1.loc[df1.range0 <= 0.0, 'isUp'] = 0#dn
        if size_output == 3 :
            df1['isUp'] = 2#hold
            df1.loc[df1.percentage >= +0.1, 'isUp'] = 1#up
            df1.loc[df1.percentage <= -0.1, 'isUp'] = 0#dn
            # df1.loc[(-0.1 < df1.percentage <  +0.1), 'isUp'] =  0
    shift =-1 #-1

    df1['isNextBarUp'] = df1['isUp'].shift(shift)# today's dataset  procuce  prediction is tommorow is up
    df1['isNextBarUp'] = df1['isNextBarUp'] .fillna(0)#.astype(int)#https://github.com/pylablanche/gcForest/issues/2
    df1['isPrev1Up'] = df1['isPrev1Up'] .fillna(0)#.astype(int)#https://github.com/pylablanche/gcForest/issues/2
    df1['isPrev2Up'] = df1['isPrev2Up'] .fillna(0)#.astype(int)#https://github.com/pylablanche/gcForest/issues/2
    df1['isNextBarUp'] = df1['isNextBarUp'].astype(int)
    df1['isPrev1Up'] = df1['isPrev1Up'].astype(int)
    df1['isPrev2Up'] = df1['isPrev2Up'].astype(int)
    df1['isUp'] = df1['isUp'].astype(int)
    '''
    df1=
    Date            Open        Close      range      isUp
    1964-05-01    79.459999    80.169998   0.300003   1.0
    1964-05-04    80.169998    80.470001   0.409996   1.0
    1964-05-05    80.470001    80.879997   0.180001   1.0
    1964-05-06    80.879997    81.059998   0.090004   1.0
    1964-05-07    81.059998    81.150002   0.000000   0.0
    1964-05-08    81.000000    81.000000  -0.099998   0.0
    1964-05-11    81.000000    80.900002   0.260002   1.0
    '''

        # direction = (close > close.shift()).astype(int)
        # target = direction.shift(-1).fillna(0).astype(int)
        # target.name = 'target'
        # sma10 = sma10.rename(columns={symbol: symbol+'sma10'})
        # sma20 = sma20.rename(columns={symbol: symbol+'sma20'})
        # df1 = df1.rename(columns={'Close': symbol+'Close'})
    '''
    # loss: 0.1222 - acc: 0.9000 - val_loss: 0.1211 - val_acc: 0.9364 epoch50  sma+range+close+open (range tell model the answer)
    # loss: 0.6932 - acc: 0.4860 - val_loss: 0.6932 - val_acc: 0.4969 random data >>> random results
    # loss: 0.6922 - acc: 0.5205 - val_loss: 0.6911 - val_acc: 0.5364 epoch50  sma
    # loss: 0.6923 - acc: 0.5198 - val_loss: 0.6914 - val_acc: 0.5360
    # loss: 0.6922 - acc: 0.5217 - val_loss: 0.6911 - val_acc: 0.5353
    # loss: 0.6431 - acc: 0.5846 - val_loss: 0.7373 - val_acc: 0.5364
    # loss: 0.5373 - acc: 0.7114 - val_loss: 0.6112 - val_acc: 0.6773            epoch50
    # loss: 0.5198 - acc: 0.7225 - val_loss: 0.5632 - val_acc: 0.6797            epoch100 sma+stoc+rsi
    # loss: 0.5487 - acc: 0.7079 - val_loss: 0.6115 - val_acc: 0.6740    SPY 1970 epoch100 sma+stoc+rsi+bol 1970
    # loss: 0.4112 - acc: 0.8140 - val_loss: 0.5576 - val_acc: 0.7324    SPY 1970 epoch500 nvo+mom+sma+stoc+rsi+bol 1970
    
    # loss: 0.6047 - acc: 0.6574 - val_loss: 0.6257 - val_acc: 0.6580    SPY 2000
    # loss:    nan - acc: 0.4711 - val_loss:    nan - val_acc: 0.4563    DJI 2000
    # loss:    nan - acc: 0.4906 - val_loss:    nan - val_acc: 0.4626    QQQ 2000
    
                          nvo         Open         High          Low        Close  range_sma  isUp
    Date                                                                                         
    1964-05-01    748.525452    79.459999    80.470001    79.459999    80.169998   0.001821   1.0
    1964-05-04    669.824179    80.169998    81.010002    79.870003    80.470001   0.005580   1.0
    2019-07-11  10607.754714  2999.620117  3002.330078  2988.800049  2999.909912   0.008677   1.0
    2019-07-12   9973.829690  3003.360107  3013.919922  3001.870117  3013.770020   0.010287   1.0
    '''
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:.2f}'.format
    print('columns=', df1.columns)
    print('\ndf1=\n', df1.loc[:, ['sma10', 'sma20', 'sma50', 'sma200', 'range_sma1']])
    print('\ndf1=\n', df1.loc[:, ['rsi10', 'rsi20', 'rsi50', 'rsi5', 'nvo', 'High', 'Low']])
    print('\ndf1=\n', df1.loc[:, ['stoc10', 'stoc20', 'stoc50', 'stoc200']])
    print('\ndf1=\n', df1.loc[:, ['bb_hi10', 'bb_hi20', 'bb_hi50', 'bb_hi200']])  # , 'sma4002']])
    print('\ndf1=\n', df1.loc[:, ['bb_lo10', 'bb_lo20', 'bb_lo50', 'bb_lo200']])  # , 'sma4002']])
    print('\ndf1=\n', df1.loc[:, ['rel_bol_hi10', 'rel_bol_hi20', 'rel_bol_hi50', 'rel_bol_hi200']])  # , 'sma4002']])
    print('\ndf1[ 0]=\n', df1.iloc[0])  # , 'sma4002']])
    #print('\ndf1[ 1]=\n', df1.iloc[1])  # , 'sma4002']])
    #print('\ndf1[9308]=\n', df1.iloc[9308])  # , 'sma4002']])
    #print('\ndf1[-2]=\n', df1.iloc[-2])  # , 'sma4002']])
    print('\ndf1[-1]=\n', df1.iloc[-1])  # , 'sma4002']])
    print('\ndf1=\n', df1.loc[:, [ 'Open', 'Close',  'range0', 'isPrev2Up','isPrev1Up', 'isUp', 'isNextBarUp']])

    # df = pd.DataFrame(record, columns = ['Name', 'Age', 'Stream', 'Percentage'])
    # rslt_df = df[df1['isUp'] == 1]
    # print ('\ndf1 describe direction = +1\n',rslt_df.describe())
    # rslt_df = df[df1['isUp'] == -1]
    # print ('\ndf1 describe direction = -1\n',rslt_df.describe())
    # rslt_df = df[df1['isUp'] == 0]
    # print ('\ndf1 describe direction =  0\n',rslt_df.describe())
    # # print ('\ndf1=\n',df1.loc[:, ['ema','macd','stoc', 'rsi']])
    print('\ndf11 describe=\n', df1.loc[:,
                                ['nvo', 'mom5', 'mom10', 'mom20', 'mom50',       'range_sma', 'range_sma1', 'range_sma2', 'range_sma3', 'range_sma4',
                                 # 'sma10', 'sma20', 'sma50', 'sma200', 'sma400', 'bb_hi10', 'bb_lo10',
                                 # 'bb_hi20', 'bb_lo20', 'bb_hi50', 'bb_lo50', 'bb_hi200', 'bb_lo200'
                                 'rel_bol_hi10',  'rel_bol_lo10', 'rel_bol_hi20', 'rel_bol_lo20', 'rel_bol_hi50', 'rel_bol_lo50',  'rel_bol_hi200', 'rel_bol_lo200',
                                 'rsi10', 'rsi20', 'rsi50', 'rsi5',        'stoc10', 'stoc20', 'stoc50', 'stoc200', 'isUp']].describe())

    df1 = df1.round(4)

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
def get_data_from_web2(symbol):
    start, end = '1970-01-03', '2019-07-12'  # '2007-05-02', '2016-04-11'
    data = pdr.get_data_yahoo(symbol, start, end)
    closePrice = data["Close"]
    print(closePrice)
    return closePrice


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
