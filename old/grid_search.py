# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Function to create model, required for KerasClassifier
def create_model( activation='relu', optimizer='rmsprop', loss= 'mse', init='glorot_uniform', dropout='0.2'):

    size_input = 39
    size_hidden = 15
    size_output = 2
    model = Sequential()
    model.add(Dense(12, kernel_initializer=init, activation=activation,  input_dim=8))
    model.add(Dense(8 , kernel_initializer=init, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(1 , kernel_initializer=init, activation='sigmoid'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("../files/input/pima-indians-diabetes.csv", delimiter=",")# load pima indians dataset

X = dataset[:,0:8]# all rows, 8 columns
Y = dataset[:,  8]# all rows, 8th column only

activation      = ['softmax', 'softplus', 'softsign', 'sigmoid', 'relu', 'tanh',  'hard_sigmoid', 'linear']
init            = [ 'zero'  , 'uniform', 'normal'  , 'lecun_uniform',  'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizers      = ['SGD'    , 'RMSprop' , 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
losses          = ['mse', 'mae']#'binary_crossentropy', 'categorical_crossentropy']
epochs          = [2]#, 100, 150] # default epochs=1,
batches         = [5]#, 10, 20]   #  default = none
nb_neurons      = [1,3,6,20,100]#5, 10, 20]
dropouts        = [0.0, 0.1, 0.2]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
weights         = [1, 2]#, 3, 4, 5]
param_grid = dict( #activation  = activation,
                  #init=init
                  #weight_constraint=weights,
                  #optimizer=optimizers
                  #epochs=epochs,
                  #batch_size=batches
                  #loss= losses
                    dropout = dropouts
                    )
model      = KerasClassifier(build_fn=create_model, verbose=2)
grid       = GridSearchCV   (estimator=model      , param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best score: %f using params %s" % (grid_result.best_score_, grid_result.best_params_))
means  = grid_result.cv_results_['mean_test_score']
stds   = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with params: %r" % (mean, stdev, param))