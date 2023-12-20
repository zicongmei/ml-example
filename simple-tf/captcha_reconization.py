from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from captcha_common import *
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


digit_size = 1
generator = generate_data(digit_size)

# data = generator.generate('12')
# generator.display(data)


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
    for i in range(y.shape[0]):
        yhat = np.argmax(prediction[i])
        if yhat != y[i][0]:
            error += 1
            # generator.display(X[i], 'want {}; got {}, perdition {}'.format(
            #     y[i][0], yhat, prediction[i]))

    return error * 1.0 / y.shape[0]


X_train, y_train = generator.generate_all(10000, file_path='/tmp/train.mat')
X_cv, y_cv = generator.generate_all(1000, file_path='/tmp/cv.mat')
X_test, y_test = generator.generate_all(1000, file_path='/tmp/test.mat')

# X_train, y_train = generator.load_data(file_path='/tmp/train.mat')
# X_cv, y_cv = generator.load_data(file_path='/tmp/cv.mat')
# X_test, y_test = generator.load_data(file_path='/tmp/test.mat')


def evaluate_model(neurons, epochs):
    model = train(X_train, y_train, neurons, epochs)
    err_train = calculate_error(model, X_train, y_train)
    err_cv = calculate_error(model, X_cv, y_cv)
    err_test = calculate_error(model, X_test, y_test)

    print(neurons, epochs, err_train, err_cv, err_test)


# evaluate_model([35,20],20)
# evaluate_model([25,15],40)
evaluate_model([25, 15], 3)
# evaluate_model([15,10],20)
