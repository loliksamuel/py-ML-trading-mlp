import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

class RNN:
    def __init__(self, hidden_cnt, num_cells, time_steps):
        self.hidden_cnt = hidden_cnt
        self.num_cells = num_cells
        self.time_steps = time_steps

    def rnn(self):
        data = pd.read_csv("files/input/GOOG.csv")
        data = data.drop(data.index[0])#(3750,5)
        data = data.drop(['Date', 'Close'], axis=1)
        data = np.array(data.values)#(3750,3)
        avg = list()
        for row in data:
            avg.append((float(row[-1]) + float(row[-2]))/2)
        avg = np.array(avg)
        samples = data.shape[0]
        avg = avg.reshape(samples, 1)#convert [1,2,3,4,5] to [[1],[2],[3],[4],[5]]  (3750,1)
        data = data[:,[0,1]]#convert (3750,5) to (3750,2)
        data = data.astype(float)
        data = np.hstack((data, avg))#combine data+array into (3750,3)
        data = data[::-1]# ::-1 means take everything in this dimension but backwards.
        split_size = int(0.8*len(data))
        train_data = data[:split_size]
        test_data = data[split_size:]

        X_train, Y_train, X_test, Y_test = train_data[:, :-1], train_data[:,-1], test_data[:, :-1], test_data[:,-1]

        sc1 = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled, testing_set_scaled  = sc1.fit_transform(X_train), sc1.transform(X_test)

        Y_train, Y_test = Y_train.reshape(-1, 1), Y_test.reshape(-1, 1)
        sc2 = MinMaxScaler(feature_range = (0, 1))
        training_target_scaled, testing_target_scaled = sc2.fit_transform(Y_train),sc2.transform(Y_test)

        xnew_train = list()
        ynew_train = list()
        for i in range(self.time_steps, training_target_scaled.shape[0]):#from 20 to 750
            xnew_train.append(training_set_scaled[i - self.time_steps:i, :])
            ynew_train.append(training_target_scaled[i, 0])
        xnew_train = np.array(xnew_train)
        ynew_train = np.array(ynew_train)
        xnew_train = np.reshape(xnew_train, (xnew_train.shape[0], 2*xnew_train.shape[1], 1))
        regressor = Sequential()
        regressor.add(LSTM(units = self.num_cells, return_sequences = True, input_shape = (xnew_train.shape[1], 1)))
        for i in range(self.hidden_cnt-2):
            regressor.add(LSTM(units = self.num_cells, return_sequences = True))
        regressor.add(LSTM(units = self.num_cells, return_sequences = False))
        regressor.add(Dense(units = 1))#1 means regression
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(xnew_train, ynew_train, epochs = 100, batch_size = 32)
        train_result = regressor.predict(xnew_train)
        plt.plot(ynew_train, color = 'green', label = 'original StockPrice')
        plt.plot(train_result, color = 'black', label = 'Predicted StockPrice')
        plt.title('Training')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()


        xnew_valid = list()
        ynew_valid = list()
        for i in range(self.time_steps, testing_target_scaled.shape[0]):
            xnew_valid.append(testing_set_scaled[i - self.time_steps:i, :])
            ynew_valid.append(testing_target_scaled[i, 0])
        xnew_valid = np.array(xnew_valid)
        ynew_valid = np.array(ynew_valid)
        xnew_valid = np.reshape(xnew_valid, (xnew_valid.shape[0], 2*xnew_valid.shape[1], 1))
        test_result = regressor.predict(xnew_valid)
        plt.plot(ynew_valid, color = 'green', label = 'original StockPrice')
        plt.plot(test_result, color = 'black', label = 'Predicted StockPrice')
        plt.title('Testing')
        plt.xlabel('Time')
        plt.ylabel('StockPrice')
        plt.legend()
        plt.show()
        #regression : loss: 0.0049

a = RNN(2, 30, 20)
a.rnn()