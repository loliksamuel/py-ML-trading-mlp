# Load the dataset as DataFrame
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from buld.build_models import MlpTrading_old
import matplotlib.pyplot as plt

from buld.utils import plot_stat_loss_vs_accuracy2

mlp = MlpTrading_old(symbol='^GSPC')
df_all = mlp.data_prepare(0.33, 16660, False)
sk_params = {'size_input': mlp.size_input,  'size_output':mlp.size_output, 'size_hidden':15, 'dropout':0.0, 'optimizer':'rmsprop', 'activation':'sigmoid'}
model = KerasClassifier(build_fn=mlp.model_create_mlp, **sk_params)
history = model.fit(mlp.x_train, mlp.y_train, sample_weight=None, batch_size=128, epochs=10   , verbose=1 )#validation_data=(mlp.x_test, mlp.y_test) kwargs=kwargs)

# model  = mlp.model_create_mlp(activation='softmax')#575/2575 [==============================] - 0s 25us/step - loss: 0.6850 - acc: 0.5538 - val_loss: 0.6900 - val_acc: 0.5296 ypeError: If no scoring is specified, the estimator passed should have a 'score' method. The estimator <keras.engine.sequential.Sequential object at 0x138a7fd68> does not.
# history = mlp.model_fit(model,epochs=10, verbose=1)


mlp.model_weights(model, mlp.x_test, mlp.y_test, mlp.names_input)
plot_stat_loss_vs_accuracy2(history.history)
plt.show()

score = model.score(mlp.x_test, mlp.y_test)
print(f'accuracy= {score} ')




#
#
#
# def baseline_model(inputs=4):
#     model = Sequential()
#     model.add(Dense(units=8, activation='sigmoid', input_shape=(inputs,)))
#     model.add(Dense(units=3, activation='sigmoid'))
#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     #model.fit()
#     return model
#
#
# df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
# df['label'] = load_iris().target
# X = df.iloc[:,0:4].values
# y = df.iloc[:,4].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# sk_params = {'inputs': 4}
# sk_params = {'size_input': 4 ,  'size_output':3, 'size_hidden':8, 'dropout':0.0, 'optimizer':'adam', 'activation':'sigmoid'}
# model = KerasClassifier(build_fn=mlp.model_create_mlp, **sk_params)
# history = model.fit(X_train, y_train, batch_size=4, epochs=100, verbose=0)#, validation_data=(X_test, y_test))
# mlp.model_weights(model,X_test,y_test,df.columns[:-1].tolist())
# plot_stat_loss_vs_accuracy2(history.history)
# plt.show()
# pred1= eli5.formatters.as_dataframe.explain_prediction_df(estimator=model, doc=X_test[0])
# pred2 = eli5.formatters.as_dataframe.explain_prediction_df(estimator=perm,   doc=X_test[0])
# print('\nweights=',weights)
# print('\npred1=',pred1)
# # print('\npred2=',pred2)
# 0  petal length (cm)  0.331579  0.061378
# 1   petal width (cm)  0.300000  0.067811
# 2   sepal width (cm)  0.078947  0.028828
# 3  sepal length (cm)  0.052632  0.000000
# 0  petal length (cm)  0.426316  0.083881
# 1   petal width (cm)  0.173684  0.093560
# 2   sepal width (cm)  0.031579  0.019693
# 3  sepal length (cm)  0.026316  0.016644
#
# weights=         feature  weight  std
# 0          rsi5    0.36 0.01
# 1  rel_bol_lo10    0.10 0.00
# 2     range_sma    0.06 0.00
# 3    range_sma1    0.01 0.00