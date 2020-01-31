import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('dataset.csv')
dataset.head()
X = dataset.drop(labels=['Liquid', 'Gas', 'Tar'], axis = 1)
y = dataset.drop(labels=['HDPE', 'LDPE', 'PP', 'PS', 'Tar'], axis = 1)
X.head()
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

model = Sequential()
model.add(Dense(X.shape[1], activation='relu', input_dim = X.shape[1]))
model.add(Dense(10, activation='relu'))
model.add(Dense(y.shape[1], activation ='sigmoid'))
model.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 10, verbose = 1)
y_pred = model.predict(X_test)
model.evaluate(X_test, y_test)
print(X_test)
print(y_pred)



