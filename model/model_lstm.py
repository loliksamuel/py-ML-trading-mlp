from keras import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM

from model.model_abstract import AbstractMlTradingModel
import numpy as np

class LstmTradingModel(AbstractMlTradingModel):

    def __init__(self, size_hidden, size_input, size_output, dropout=0.2, kernel_init='glorot_uniform') -> None:
        super().__init__(size_hidden, size_input, size_output, dropout, kernel_init)

    def name(self):
        return "LSTM"

    def _create(self):
        look_back = 1
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], look_back, self.size_input))
        self.x_test  = np.reshape(self.x_test , (self.x_test.shape [0], look_back, self.size_input))

        self._model = Sequential()

        self._model.add(
            LSTM(units=self.size_hidden, activation='relu', return_sequences=True,
                 input_shape=(look_back, self.size_input)))
        self._model.add(Dropout(self.dropout))

        self._model.add(LSTM(units=self.size_hidden, activation='relu', return_sequences=True))
        self._model.add(Dropout(self.dropout))

        self._model.add(LSTM(units=self.size_hidden, activation='relu', return_sequences=False))
        self._model.add(Dropout(self.dropout))

        self._model.add(Dense(units=self.size_output, activation='softmax'))
        self._model.summary()

    def create_2(self):
        # input_shape = (input_length, input_dim)
        # input_shape=(self.size_input,)   equals to    input_dim = self.size_input
        # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        self._model = Sequential()
        self._model.add(Embedding(input_dim=self.size_input, output_dim=self.size_hidden, input_length=self.size_input))
        # model.add(Embedding( input_shape=(self.size_input,size_hidden), kernel_initializer=kernel_init))
        self._model.add(
            LSTM(units=self.size_hidden, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True))
        self._model.add(Dropout(self.dropout))
        self._model.add(LSTM(units=self.size_hidden, activation='sigmoid', inner_activation='hard_sigmoid'))
        self._model.add(Dropout(self.dropout))
        self._model.add(Dense(self.size_output, activation='softmax'))
        self._model.summary()
