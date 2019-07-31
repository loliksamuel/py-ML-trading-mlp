from model.model_lstm import LstmTradingModel
from model.enum import MlModel
from model.model_mlp import MlpTradingModel
from buld.singleton import Singleton


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
