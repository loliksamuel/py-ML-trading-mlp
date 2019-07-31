from keras.optimizers import RMSprop
from pylab import *
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

from buld.utils import plot_stat_loss_vs_accuracy2


def baseline_model(lr=0.01):
    model = Sequential()
    model.add(Dense(1, activation = 'linear', input_dim = 1))
    model.compile(  optimizer=RMSprop(lr=lr)
                  , loss = 'mean_squared_error'
                  , metrics = ['accuracy'])
    return model


epochs = 150#for better accuracy increase epochs or lr
lr     = 0.01
batch  = 32
data   = data = linspace(1,2,100).reshape(-1,1)
y = data*10+5
#Use the model
regr = baseline_model(lr=lr)
history = regr.fit(data,y, epochs=epochs, batch_size=batch)

print('\nplot_accuracy_loss_vs_time...')
history_dict = history.history

plt.clf()
plt.plot(data, regr.predict(data), 'b', data,y, 'k.')
plt.show()

#plt.clf()
plot_stat_loss_vs_accuracy2(history_dict)
#
plt.show()