import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from func import plot_input_img, test_model

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Pre - processing

X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Model Set-up
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10,activation = 'softmax'))
model.compile(optimizer = 'adam',loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])

# Callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 4, verbose = 1)

mc = ModelCheckpoint("./bestmodel.keras", monitor = 'val_accuracy', verbose = 1, save_best_only = True)

cb = [es,mc]

# Model Training
his = model.fit(X_train, y_train, epochs = 50, validation_split = 0.2, callbacks = cb)

# Test

score = test_model(X_test,y_test)
print(f" Accuracy is {score}")