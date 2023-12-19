import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
import warnings
import random

from number_classification_common import *


X, y = load_data()

X, y = shuffle(X, y)

# plot_X(X, y)

def split_data(X, y):
    train_end = int(X.shape[0] * 6 / 10)
    cv_end = int(X.shape[0] * 8 / 10)
    test_end = X.shape[0]
    # print(X.shape[0],train_end)
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_cv = X[train_end:cv_end]
    y_cv = y[train_end:cv_end]
    X_test = X[cv_end:test_end]
    y_test = y[cv_end:test_end]

    return X_train, y_train, X_cv, y_cv, X_test, y_test

X_train, y_train, X_cv, y_cv, X_test, y_test = split_data(X, y)


def train(X, y, neurons, epochs):
    model = Sequential(
        [               
            tf.keras.Input(shape=(X.shape[1],)),  
            Dense(neurons[0], activation="relu", name="layer1"),
            Dense(neurons[1], activation="relu", name="layer2"),
            Dense(10, activation="linear", name="layer3"),
            
        ], name = "my_model" 
    )                            
    # model.summary()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    model.fit(
        X,y,
        epochs=epochs,
        verbose=0
    )
    return model

def calculate_error(model, X, y):
    prediction = model.predict(X,verbose=0)
    
    error = 0
    err_X = np.zeros([64, X.shape[1]])
    err_y = np.zeros([64, y.shape[1]])
    for i in range(y.shape[0]):
        yhat = np.argmax(prediction[i])
        if yhat != y[i][0]:
            if error < 64:
                err_X[error] = X[i]
                err_y[error][0] = yhat
            error += 1

    # plot_X(err_X, err_y)
    return error * 1.0 / y.shape[0]

def evaluate_model(neurons, epochs):
    model = train(X_train, y_train,neurons, epochs)
    err_train = calculate_error(model, X_train, y_train)
    err_cv = calculate_error(model, X_cv, y_cv)
    err_test = calculate_error(model, X_test, y_test)

    print(neurons, epochs, err_train, err_cv, err_test )


# evaluate_model([35,20],20)
# evaluate_model([25,15],40)
evaluate_model([25,15],20)
# evaluate_model([15,10],20)

