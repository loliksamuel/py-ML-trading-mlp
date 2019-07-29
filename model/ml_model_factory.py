from model.lstm_trading_model import LstmTradingModel
from model.ml_model import MlModel
from model.mlp_trading_model import MlpTradingModel
from utils.singleton import Singleton


class MlModelFactory(object):
    __metaclass__ = Singleton

    def __init__(self) -> None:
        super().__init__()
        self.models = dict()
        self.models[MlModel.MLP]  = MlpTradingModel
        self.models[MlModel.LSTM] = LstmTradingModel

    def create(self, model_type, size_hidden, size_input, size_output, dropout=0.2, kernel_init='glorot_uniform'):
        return self.models[model_type](size_hidden=size_hidden, size_input=size_input, size_output=size_output,
                                       dropout=dropout, kernel_init=kernel_init)
