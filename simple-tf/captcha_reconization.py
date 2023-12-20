from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from captcha_common import *
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

generator = generate_data()

# data = generator.generate('12')
# generator.display(data)


digit_size = 1


def generate_data(data_size):
    X = np.empty([data_size, 60*160*3])
    y = np.empty([data_size, 1])

    for i in range(data_size):
        print("generating data #", i)
        y[i][0] = random.randint(0, 10**digit_size - 1)
        str_y = str(y)
        if len(str_y) == 1:
            str_y = '0' + str_y
        X[i] = generator.generate('12')
    return X, y


def train(X, y, neurons, epochs):
    model = Sequential(
        [
            tf.keras.Input(shape=(X.shape[1],)),
            Dense(neurons[0], activation="relu", name="layer1"),
            Dense(neurons[1], activation="relu", name="layer2"),
            Dense(10**digit_size, activation="linear", name="layer3"),

        ], name="my_model"
    )
    # model.summary()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    model.fit(
        X, y,
        epochs=epochs,
    )
    return model


def calculate_error(model, X, y):
    prediction = model.predict(X)

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


X_train, y_train = generate_data(10000)
X_cv, y_cv = generate_data(1000)
X_test, y_test = generate_data(1000)


def evaluate_model(neurons, epochs):
    model = train(X_train, y_train, neurons, epochs)
    err_train = calculate_error(model, X_train, y_train)
    err_cv = calculate_error(model, X_cv, y_cv)
    err_test = calculate_error(model, X_test, y_test)

    print(neurons, epochs, err_train, err_cv, err_test)


# evaluate_model([35,20],20)
# evaluate_model([25,15],40)
evaluate_model([25, 15], 20)
# evaluate_model([15,10],20)
