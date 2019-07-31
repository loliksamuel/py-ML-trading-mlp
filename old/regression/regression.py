from keras.optimizers import RMSprop, SGD
from keras.wrappers.scikit_learn import KerasRegressor
from pylab import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

from buld.utils import plot_stat_loss_vs_accuracy2


def gridSearch_neural_network():
    # fix random seed for reproducibility
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=build_model, nb_epoch=100, batch_size=5, verbose=0)

    # grid search epochs, batch size and optimizer
    optimizers   = ['SGD'    , 'RMSprop' , 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    dropout_rate = [0.0, 0.1, 0.2]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    init         = ['glorot_uniform', 'normal', 'uniform']
    epochs       = [50, 100, 150]
    batches      = [20]#5, 10, 20]
    weight_constraint = [1, 2]#, 3, 4, 5]
    param_grid = dict(optimizer   = optimizers,
                      dropout_rate= dropout_rate,
                      activation  = activation,
                      epochs      = epochs,
                      batch_size  = batches,
                      weight_constraint=weight_constraint,
                      init=init)

    grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
    grid_result = grid.fit(x, y)
    # summarize results
    print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    means  = grid_result.cv_results_["mean_test_score"]
    stds   = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('%f (%f) with: %r' % (mean, stdev, param))


def build_model(x, y, f_optimizer=SGD(lr=0.001), f_loss='mean_squared_error', f_metrics = ['accuracy'], f_activation='relu', pct_dropout=0.2, nb_neurons=10, nb_layers=2, nb_epochs=10, nb_batch=10):
    model = Sequential()

    model.add(Dense  (units=nb_neurons, activation=f_activation, init='uniform', input_dim = 1))

    for _ in range(nb_layers):
        model.add(Dense  (units=nb_neurons, activation=f_activation, init='uniform'))
        model.add(Dropout(pct_dropout))  # for generalization

    model.add(Dense(1, activation = 'linear'))#,  init='uniform' ))
    #model.add(Dense(1, activation = 'linear', input_dim = 1))

    model.summary()
    model.compile                (optimizer=f_optimizer , loss = f_loss     , metrics = f_metrics)
    history     = model.fit(x, y, epochs=nb_epochs      , batch_size=nb_batch,verbose=2)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x, model.predict(x), 'b', x, y, 'r')
    plt.title('true vs prediction regression')
    plt.legend()
    return history

seed = 7
np.random.seed(seed)
# 100 numbers from 1 to 2
x           = linspace(1,2,100).reshape(-1,1)
p           = 5# 1 will be solved no problem with nb_layers   = 0 , nb_neurons  = 2, nb_batch    = 10, nb_epochs    = 400
y           = (x**p)*10+5

lr          = 0.001
f_optimizer = SGD(lr=lr)#    optimizers   = ['SGD'    , 'RMSprop' , 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
f_loss        ='mse'#mse 'mean_squared_error'
f_metrics     = ['accuracy']
f_activation = 'relu' # ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

nb_neurons  = 20
nb_epochs   = 400#for better accuracy increase epochs or lr
nb_batch    = 10
nb_layers   = 0# do not change that
pct_dropout = 0.2
history     = build_model(   x
                           , y
                           , pct_dropout=pct_dropout
                           , nb_neurons=nb_neurons
                           , nb_layers=nb_layers
                           , nb_epochs=nb_epochs
                           , nb_batch=nb_batch
                           , f_activation = f_activation
                           , f_optimizer=f_optimizer
                           , f_loss    = f_loss
                           , f_metrics = f_metrics  )

#model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
#print (x)
#print ('y=',y)
print('\nplot_accuracy_loss_vs_time...')
#plt.clf()
plt.subplot(1, 2, 2)
plot_stat_loss_vs_accuracy2(history.history)
plt.show()

