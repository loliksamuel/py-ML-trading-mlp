from keras import Sequential
from keras.layers import Dense, Dropout

from model.abstract_ml_trading_model import AbstractMlTradingModel


class MlpTradingModel(AbstractMlTradingModel):

    def __init__(self, size_hidden, size_input, size_output, dropout=0.2, kernel_init='glorot_uniform') -> None:
        super().__init__(size_hidden, size_input, size_output, dropout, kernel_init)

    def name(self):
        return "MLP"

    def _create(self):
        self._model = Sequential()  # stack of layers

        self._model.add(
            Dense(units=self.size_hidden, activation='relu', input_shape=(self.size_input,),
                  kernel_initializer=self.kernel_init))
        self._model.add(Dropout(self.dropout))  # for generalization

        self._model.add(Dense(units=self.size_hidden, activation='relu'))
        self._model.add(Dropout(self.dropout))  # for generalization.

        self._model.add(Dense(units=self.size_hidden, activation='relu'))
        self._model.add(Dropout(self.dropout))  # regularization technique by removing some nodes

        self._model.add(Dense(units=self.size_output, activation='softmax'))
        self._model.summary()
