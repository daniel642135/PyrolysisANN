# import tensorflow as tf
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
y = dataset.drop(labels=['HDPE', 'LDPE', 'PP', 'PS', 'Tar'], axis = 1)
X.head()
y.head()
# Dimensions of dataset
n = X.shape[0]
p = X.shape[1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Scale data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def create_model():
    model = Sequential()
    model.add(Dense(X.shape[1], activation='sigmoid', input_dim = X.shape[1]))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(y.shape[1], activation ='sigmoid'))
    model.compile(optimizer='SGD', loss = 'mse', metrics=['accuracy'])
    return model

model = create_model()
model.summary()
earlystop_callback = EarlyStopping(
  monitor='accuracy', min_delta=0.0001,
  patience=1)
model.fit(X_train, y_train.to_numpy(), epochs = 100, verbose = 1, callbacks=[earlystop_callback])
model.save_weights('./checkpoints/my_checkpoints')
y_pred = model.predict(X_test)

model = create_model()
# model.load_weights('./checkpoints/my_checkpoints')
loss, accuracy = model.evaluate(X_test, y_test.to_numpy())
print("Accuracy = {:5.2f}%".format(100*accuracy))