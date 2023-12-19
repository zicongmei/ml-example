import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import logging
import scipy.io
import urllib.request
import warnings
from number_classification_common import *

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_all, y_all = load_data()

# print ('The first element of X is: ', X[0])
print('The shape of X is: ' + str(X_all.shape))
print('The shape of y is: ' + str(y_all.shape))

# plot_random_X(X_all, y_all)


def identify_one_digit(X_all, y_all, target):
    X = X_all
    y = y_all.copy()
    for i in range(y.shape[0]):
        if y[i][0] == target:
            y[i][0] = 1
        else:
            y[i][0] = 0
    # plot_X(X, y)
    # break
    model = Sequential(
        [
            tf.keras.Input(shape=(X.shape[1],)),  # specify input size
            Dense(25, activation="sigmoid", name="layer1"),
            Dense(15, activation="sigmoid", name="layer2"),
            Dense(1, activation="sigmoid", name="layer3"),
        ], name="my_model"
    )
    model.summary()
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )
    model.fit(
        X, y,
        epochs=20
    )
    result = model.predict(X)
    # for i in range(result.shape[0]):
    #     if result[i][0] < 0.5:
    #         result[i][0] = 0
    #     else:
    #         result[i][0] = 1
    plot_random_X(X_all, result)


identify_one_digit(X_all, y_all, 3)
identify_one_digit(X_all, y_all, 6)
