# Load the dataset as DataFrame
import pandas as pd
from sklearn.datasets import load_iris
df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
df['label'] = load_iris().target

# Divide to X and y and split to train and test sets
X = df.iloc[:,0:4].values
y = df.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Define the neural network model
from keras.models import Sequential
from keras.layers import Dense
def baseline_model():
    model = Sequential()
    model.add(Dense(units=8, activation='relu', input_dim=4))
    model.add(Dense(units=3, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model
from keras.wrappers.scikit_learn import KerasClassifier
my_model = KerasClassifier(build_fn=baseline_model)
my_model.fit(X_train, y_train, batch_size=4, epochs=100, verbose=0)

# Run eli5 explanations
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(my_model, random_state=1).fit(X_test, y_test)

# This actually works!
explanation = eli5.formatters.as_dataframe.explain_weights_df(perm,
                                                              feature_names=df.columns[:-1].tolist())

# But this line does not "work", i.e. it produces a NoneType object
explanation_pred = eli5.formatters.as_dataframe.explain_prediction_df(estimator=my_model,
                                                   doc=X_test[0])