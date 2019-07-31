import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from buld.utils import plot_stat_loss_vs_time, plot_stat_accuracy_vs_time, plot_stat_loss_vs_accuracy


class LSTM:
    def __init__(self, hidden_cnt, num_cells, time_steps):
        self.hidden_cnt = hidden_cnt
        self.num_cells = num_cells
        self.time_steps = time_steps

    def run(self):
        data = pd.read_csv("../../files/input/GOOG.csv")
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

        #model regressor
        model = Sequential()
        model.add(LSTM(units=self.num_cells, return_sequences=True, input_shape=(xnew_train.shape[1], 1)))
        for i in range(self.hidden_cnt-2):
            model.add(LSTM(units=self.num_cells, return_sequences=True))
        model.add(LSTM (units=self.num_cells, return_sequences=False))
        model.add(Dense(units=1))#1 means regression
        model.compile  (optimizer='adam', loss='mean_squared_error')

        history = model.fit(xnew_train, ynew_train, epochs = 100, batch_size = 32)

        history_dict = history.history
        print(history_dict.keys())
        plot_stat_loss_vs_time     (history_dict)
        plot_stat_accuracy_vs_time (history_dict)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist.tail())

        plot_stat_loss_vs_accuracy(history_dict)
        score = model.evaluate(testing_set_scaled, testing_target_scaled, verbose=1)                                     # random                                |  calc label
        print('Test loss:    ', score[0], ' (is it close to 0?)')                            #Test,train loss     : 0.6938 , 0.6933   |  0.47  0.5
        print('Test accuracy:', score[1], ' (is it close to 1 and close to train accuracy?)')#Test,train accuracy : 0.5000 , 0.5000   |  0.69, 0.74


        train_result = model.predict(xnew_train)
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
        test_result = model.predict(xnew_valid)
        plt.plot(ynew_valid , color = 'green', label = 'original StockPrice')
        plt.plot(test_result, color = 'black', label = 'Predicted StockPrice')
        plt.title('Testing')
        plt.xlabel('Time')
        plt.ylabel('StockPrice')
        plt.legend()
        plt.show()
        #regression : loss: 0.0049

lstm = LSTM(hidden_cnt=2, num_cells=3, time_steps=1)#hidden_cnt, num_cells, time_steps):
lstm.run()