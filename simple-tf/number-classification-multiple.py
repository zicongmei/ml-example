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

X, y = load_data()

# print ('The first element of X is: ', X[0])
print('The shape of X is: ' + str(X.shape))
print('The shape of y is: ' + str(y.shape))

plot_random_X(X, y)


model = Sequential(
    [
        tf.keras.Input(shape=(X.shape[1],)),
        Dense(25, activation="relu", name="layer1"),
        Dense(15, activation="relu", name="layer2"),
        Dense(10, activation="linear", name="layer3"),
    ], name="my_model"
)
model.summary()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
model.fit(
    X, y,
    epochs=20
)

prediction = model.predict(X)
prediction_p = tf.nn.softmax(prediction)

yhat = np.empty([y.shape[0], 1])
m, n = X.shape
for i in range(y.shape[0]):
    yhat[i][0] = np.argmax(prediction_p[i])

plot_random_X(X, yhat)
