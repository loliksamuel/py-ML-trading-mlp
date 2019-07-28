# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    # create model

    size_input = 39
    size_hidden = 15
    size_output = 2
    model = Sequential()
    #model.add(Dense(size_hidden , activation='relu', input_dim=8, kernel_initializer=init ))
    model.add(Dense  (size_hidden, activation='relu', input_shape=(size_input,), kernel_initializer=init))
    model.add(Dropout(0.2))#for generalization
    # model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense  (size_hidden, activation='relu'))
    model.add(Dropout(0.2))#for generalization
    model.add(Dense  (size_output, activation='softmax'))# last layer always has softmax(except for regession problems and  binary- 2 classes where sigmoid is enough)

    # model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam', 'adagrad']
init       = ['glorot_uniform', 'normal', 'uniform']# kernel_initializer='glorot_uniform', 'normal', 'uniform'
epochs     = [10, 15, 20, 50, 128]
batches    = [ 5, 10, 128]
param_grid = dict  (optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model     , param_grid=param_grid)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means  = grid_result.cv_results_['mean_test_score']
stds   = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))