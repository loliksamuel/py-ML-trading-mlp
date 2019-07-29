import numpy as np
import pandas as pd
from abc import abstractmethod
from keras.optimizers import RMSprop

from utils.utils import plot_stat_loss_vs_time, plot_stat_accuracy_vs_time, plot_stat_loss_vs_accuracy, plot_conf_mtx


class AbstractMlTradingModel(object):

    def __init__(self, size_hidden, size_input, size_output, dropout=0.2, kernel_init='glorot_uniform') -> None:
        super().__init__()
        self.size_hidden = size_hidden
        self.size_input = size_input
        self.size_output = size_output
        self.dropout = dropout
        self.kernel_init = kernel_init
        self._model = None
        self._history = None
        self._create()

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _create(self):
        pass

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def compile(self, loss='categorical_crossentropy', lr=0.00001, rho=0.9, epsilon=None, decay=0.0):
        self._model.compile(loss=loss,  # measure how accurate the model during training
                            optimizer=RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay),
                            # this is how model is updated based on data and loss function
                            metrics=['accuracy'])

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size, verbose=0):
        self._history = self._model.fit(x_train,
                                        y_train,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        validation_data=(x_test, y_test),
                                        # validation_split = 0.1,
                                        verbose=verbose)

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def evaluate(self, x_test, y_test):
        score = self._model.evaluate(x_test, y_test, verbose=0)
        print(f'Total loss:    {score[0]} (is it close to 0 ?)')
        print(f'Total accuracy:{score[1]} (is it close to 1 and close to train accuracy ?)')
        return score

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def plot_evaluation(self, size_input, size_output, iteration_id, title=''):
        print(f'\nsize.model.features(size_input) = {size_input}')
        print(f'\nsize.model.target  (size_output)= {size_output}')

        print('\nplot_accuracy_loss_vs_time...')
        history_dict = self._history.history
        print(history_dict.keys())

        #plot_stat_loss_vs_time(history_dict, title=f'{iteration_id}model Loss over time{title}')
        #plot_stat_accuracy_vs_time(history_dict, title=f'{iteration_id}model Accuracy over time{title}')
        plot_stat_loss_vs_accuracy(history_dict, title=f'{iteration_id}model Loss, Accuracy over time{title}')

        hist = pd.DataFrame(self._history.history)
        hist['epoch'] = self._history.epoch
        print(hist.tail())

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def predict(self,  x_train, y_train, x_test, y_test, names_output=[0,1], iteration_id=''):
        y_train_pred = self._model.predict(x_train)
        print(f'labeled   as {y_test[0]} highest confidence for {np.argmax(y_test[0])}')
        print(f'predicted as {y_train_pred[0]} highest confidence for {np.argmax(y_train_pred[0])}')
        Y_true = np.argmax(y_train, axis=1)
        Y_pred = np.argmax(y_train_pred, axis=1)
        plot_conf_mtx(Y_true, Y_pred,  names_output, f'files/output/{iteration_id}Confusion matrix on train.png')

        y_test_pred = self._model.predict(x_test)
        print(f'labeled   as {y_test[0]} highest confidence for {np.argmax(y_test[0])}')
        print(f'predicted as {y_test_pred[0]} highest confidence for {np.argmax(y_test_pred[0])}')
        Y_true = np.argmax(y_test, axis=1)
        Y_pred = np.argmax(y_test_pred, axis=1)
        plot_conf_mtx(Y_true, Y_pred,  names_output, f'files/output/{iteration_id}Confusion matrix on test.png')

        x_all = np.concatenate((x_train, x_test), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)
        y_pred = self._model.predict(x_all)
        print(f'labeled   as {y_all[0]} highest confidence for {np.argmax(y_all[0])}')
        print(f'predicted as {y_pred[0]} highest confidence for {np.argmax(y_pred[0])}')
        Y_true = np.argmax(y_all, axis=1)
        Y_pred = np.argmax(y_pred, axis=1)
        plot_conf_mtx(Y_true, Y_pred,  names_output, f'files/output/{iteration_id}Confusion matrix on all.png')
    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def summary(self):
        self._model.summary()

    # |--------------------------------------------------------|
    # |                                                        |
    # |--------------------------------------------------------|
    def save(self, folder, filename, iteration_id=''):
        print(f'\nSave model as {folder}{iteration_id}model{filename}.model')
        self._model.save(f'{folder}{iteration_id}model{filename}.model')
