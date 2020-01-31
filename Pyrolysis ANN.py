import tensorflow as tf

# import math
# import numpy as np
# from tensorflow import keras
# import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping


import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataset.csv')
dataset.head()
X = dataset.drop(labels=['Liquid', 'Gas', 'Tar'], axis = 1)
y = dataset.drop(labels=['HDPE', 'LDPE', 'PP', 'PS'], axis = 1)
X.head()
y.head()
# Dimensions of dataset
n = X.shape[0]
p = X.shape[1]

# random_state = integer will seed the randomization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)
# Scale data
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
X_train = X_train/100
X_test = X_test/100
y_train = y_train/100
y_test = y_test/100


def create_model():
    model = Sequential()
    model.add(Dense(X.shape[1], activation='sigmoid', input_dim = X.shape[1]))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(y.shape[1], activation ='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=optimizer, loss = 'mse', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

earlystop_callback = EarlyStopping(
  monitor='loss', min_delta=0.0001,
  patience=10)

#attempt to plot but there is no tensorflow_docs module
history = model.fit(X_train, y_train.to_numpy(), epochs = 1000, verbose = 1, callbacks=[earlystop_callback])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


model.save_weights('./checkpoints/my_checkpoints')
y_pred = model.predict(X_test)
print(y_pred)
print(y_test.to_numpy)
# attempt to reset the model by calling the function again, but the weights seemed to be still recorded.
# model = create_model()
# model.load_weights('./checkpoints/my_checkpoints')

loss, accuracy = model.evaluate(X_test, y_test.to_numpy(), verbose = 1)
print("Accuracy = {:5.2f}%".format(100*accuracy))