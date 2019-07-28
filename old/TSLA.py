import numpy as np
import pickle
import json
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt



def test_run():

    # Define date range
    start_date='2018-05-21'
    end_date  ='2019-05-26'
    dates=pd.date_range(start_date,end_date)
    print("dates="  ,dates)
    print("date[0]=",dates[0])

    # Define stock symbols
    symbols = ['TSLA', 'GOOG', 'FB']  # SPY will be added in get_data()

    # Get stock data
    df = get_data(symbols, dates)

    # Slice and plot
    plot_selected(df, symbols, start_date, end_date, shouldNormalize=True)

    # Slice and plot
    plot_selected(df, symbols, start_date, end_date, shouldNormalize=False)

    df_TSLA = pd.read_csv('TSLA.csv', index_col="Date", parse_dates=True, usecols=["Date","Close","High","Low"], na_values=["nan"])
    df_GOOG = pd.read_csv('GOOG.csv', index_col="Date", parse_dates=True, usecols=["Date","Close","High","Low"], na_values=["nan"])
    # print(df_TSLA.head())
    # print(df_GOOG.head())
    # plt.plot(df_TSLA['Low'  ], label='tsla low')
    # plt.plot(df_TSLA['High' ], label='tsla high')
    # plt.plot(df_GOOG['Close'], label='goog close')
    # plt.plot(df_GOOG['High' ], label='goog high')
    # plt.title('stock price (not normalized)')
    # plt.xlabel('date')
    # plt.ylabel('price')
    # plot_data(df, title='stock price (not normalized)')
   # plt.axes([2,3,4])
    #df_crosscorrelated.plot(x='Date',y=['Close','ma7','ma14','ma25'],ax=ax)
    # plt.legend()
    # plt.show()


    df_DATES = pd.DataFrame(index=dates)

    print ("df_dates=")
    print (df_DATES)
    #
    # print ("df_join=")
    # df_TSLA = df_TSLA.rename(columns={'High':'H_TSLA'})
    # df_j1 = df_TSLA.join(df_GOOG, how='inner')
    # print (df_j1)

    print ("df_join=")
    # df_TSLA = df_TSLA.rename(columns={'High':'H_TSLA'})
    df_j2 = df_DATES.join(df_TSLA, how='inner')
    df_j2  = df_j2.dropna()
    print (df_j2)

def plot_selected(df, columns, start_index, end_index, shouldNormalize = True):
    """Plot the desired columns over index values in the given range."""
    #df = df[columns][start_index:end_index]
    df.ix[start_index:end_index, columns]
    if shouldNormalize:
        df = normalize(df)
    plot_data(df, title='stock price (normalized)')

# normalize to first row
def normalize(df):
    return df/df.ix[0,:]

def symbol_to_path(symbol, base_dir=""):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def kpi_sharpeRatio():

    risk_free_rate = 2.25 # 10 year US-treasury rate (annual) or 0
    sharpe = 2
    #  ((mean_daily_returns[stocks[0]] * 100 * 252) -  risk_free_rate ) / (std[stocks[0]] * 100 * np.sqrt(252))
    return sharpe

def loss_log():
    return 2

def loss_mse():
    return 2


def loss_gdc():
    return 2

def activation_sigmoid():
    return 2

def kpi_commulativeReturn():
    return 2


def kpi_risk(df):
    return df.std()


def kpi_sharpeRatio():
    return 2




def plot_data(df, title="stock price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def get_data(symbols, dates):
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



def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

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


class Deep_Evolution_Strategy:

    inputs = None

    def __init__(
        self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights
    
    def train(self, epoch = 100, print_every = 1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')
        
class Model:
    
    DROPOUT = 0.9
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.normal(scale = 0.05, size = (input_size, layer_size)),
            np.random.normal(scale = 0.05, size = (layer_size, layer_size)),
            np.random.normal(scale = 0.05, size = (layer_size, output_size)),
            np.zeros((1, layer_size)),
            np.zeros((1, layer_size)),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-2]
        feed = np.dot(feed, self.weights[1]) + self.weights[-1]
        decision = np.dot(feed, self.weights[2])
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights


class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, timeseries, skip, initial_money, real_trend, minmax):
        self.model = model
        self.timeseries = timeseries
        self.skip = skip
        self.real_trend = real_trend
        self.initial_money = initial_money
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )
        self.minmax = minmax
        self._initiate()

    def _initiate(self):
        # i assume first index is the close value
        self.trend = self.timeseries[0]
        self._mean = np.mean(self.trend)
        self._std = np.std(self.trend)
        self._inventory = []
        self._capital = self.initial_money
        self._queue = []
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]

    def reset_capital(self, capital):
        if capital:
            self._capital = capital
        self._scaled_capital = self.minmax.transform([[self._capital, 2]])[0, 0]
        self._queue = []
        self._inventory = []

    def trade(self, data):
        """
        you need to make sure the data is [close, volume]
        """
        scaled_data = self.minmax.transform([data])[0]
        real_close = data[0]
        close = scaled_data[0]
        if len(self._queue) >= window_size:
            self._queue.pop(0)
        self._queue.append(scaled_data)
        if len(self._queue) < window_size:
            return {
                'status': 'data not enough to trade',
                'action': 'fail',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
            }
        state = self.get_state(
            window_size - 1,
            self._inventory,
            self._scaled_capital,
            timeseries = np.array(self._queue).T.tolist(),
        )
        action, prob = self.act_softmax(state)
        print(prob)
        if action == 1 and self._scaled_capital >= close:
            self._inventory.append(close)
            self._scaled_capital -= close
            self._capital -= real_close
            return {
                'status': 'buy 1 unit, cost %f' % (real_close),
                'action': 'buy',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
            }
        elif action == 2 and len(self._inventory):
            bought_price = self._inventory.pop(0)
            self._scaled_capital += close
            self._capital += real_close
            scaled_bought_price = self.minmax.inverse_transform(
                [[bought_price, 2]]
            )[0, 0]
            try:
                invest = (
                    (real_close - scaled_bought_price) / scaled_bought_price
                ) * 100
            except:
                invest = 0
            return {
                'status': 'sell 1 unit, price %f' % (real_close),
                'investment': invest,
                'gain': real_close - scaled_bought_price,
                'balance': self._capital,
                'action': 'sell',
                'timestamp': str(datetime.now()),
            }
        else:
            return {
                'status': 'do nothing',
                'action': 'nothing',
                'balance': self._capital,
                'timestamp': str(datetime.now()),
            }

    def change_data(self, timeseries, skip, initial_money, real_trend, minmax):
        self.timeseries = timeseries
        self.skip = skip
        self.initial_money = initial_money
        self.real_trend = real_trend
        self.minmax = minmax
        self._initiate()

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0])

    def act_softmax(self, sequence):
        decision = self.model.predict(np.array(sequence))

        return np.argmax(decision[0]), softmax(decision)[0]

    def get_state(self, t, inventory, capital, timeseries):
        state = get_state(timeseries, t)
        len_inventory = len(inventory)
        if len_inventory:
            mean_inventory = np.mean(inventory)
        else:
            mean_inventory = 0
        z_inventory = (mean_inventory - self._mean) / self._std
        z_capital = (capital - self._mean) / self._std
        concat_parameters = np.concatenate(
            [state, [[len_inventory, z_inventory, z_capital]]], axis = 1
        )
        return concat_parameters

    def get_reward(self, weights):
        initial_money = self._scaled_capital
        starting_money = initial_money
        invests = []
        self.model.weights = weights
        inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)

        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t]

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                starting_money += self.trend[t]
                invest = ((self.trend[t] - bought_price) / bought_price) * 100
                invests.append(invest)

            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )
        invests = np.mean(invests)
        if np.isnan(invests):
            invests = 0
        score = (starting_money - initial_money) / initial_money * 100
        return invests * 0.7 + score * 0.3

    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every = checkpoint)

    def buy(self):
        initial_money = self._scaled_capital
        starting_money = initial_money

        real_initial_money = self.initial_money
        real_starting_money = self.initial_money
        inventory = []
        real_inventory = []
        state = self.get_state(0, inventory, starting_money, self.timeseries)
        states_sell = []
        states_buy = []

        for t in range(0, len(self.trend) - 1, self.skip):
            action, prob = self.act_softmax(state)
            print(t, prob)

            if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - 1 - window_size):
                inventory.append(self.trend[t])
                real_inventory.append(self.real_trend[t])
                real_starting_money -= self.real_trend[t]
                starting_money -= self.trend[t]
                states_buy.append(t)
                print(
                    'day %d: buy 1 unit at price %f, total balance %f'
                    % (t, self.real_trend[t], real_starting_money)
                )

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                real_bought_price = real_inventory.pop(0)
                starting_money += self.trend[t]
                real_starting_money += self.real_trend[t]
                states_sell.append(t)
                try:
                    invest = (
                        (self.real_trend[t] - real_bought_price)
                        / real_bought_price
                    ) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, self.real_trend[t], invest, real_starting_money)
                )
            state = self.get_state(
                t + 1, inventory, starting_money, self.timeseries
            )

        invest = (
            (real_starting_money - real_initial_money) / real_initial_money
        ) * 100
        total_gains = real_starting_money - real_initial_money
        return states_buy, states_sell, total_gains, invest


#
#
#
# @app.route('/', methods = ['GET'])
# def hello():
#     return jsonify({'status': 'OK'})
#
#
# @app.route('/inventory', methods = ['GET'])
# def inventory():
#     return jsonify(agent._inventory)
#
#
# @app.route('/queue', methods = ['GET'])
# def queue():
#     return jsonify(agent._queue)
#
#
# @app.route('/balance', methods = ['GET'])
# def balance():
#     return jsonify(agent._capital)
#
#
# @app.route('/trade', methods = ['GET'])
# def trade():
#     data = json.loads(request.args.get('data'))
#     return jsonify(agent.trade(data))
#
#
# @app.route('/reset', methods = ['GET'])
# def reset():
#     money = json.loads(request.args.get('money'))
#     agent.reset_capital(money)
#     return jsonify(True)
#

if __name__ == '__main__':
    # app.run(host = '0.0.0.0', port = 8005)
    test_run()
