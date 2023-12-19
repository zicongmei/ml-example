import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import urllib.request
import random


def plot_X(X, y):
    m, n = X.shape
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1)
    for i, ax in enumerate(axes.flat):
        X_random_reshaped = X[i].reshape((20, 20)).T
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        # Display the label above the image
        ax.set_title(round(y[i, 0], 2))
        ax.set_axis_off()
    plt.show()


def plot_random_X(X, y):
    size = 64
    X_out = np.empty([size, X.shape[1]])
    y_out = np.empty([size, y.shape[1]])
    m, n = X.shape
    for i in range(size):
        # Select random indices
        random_index = np.random.randint(m)
        X_out[i] = X[random_index]
        y_out[i] = y[random_index]
    plot_X(X_out, y_out)


def load_data():
    data_url = 'https://raw.githubusercontent.com/kaleko/CourseraML/master/ex3/data/ex3data1.mat'
    data_local = "/tmp/ex3data1.mat"
    urllib.request.urlretrieve(data_url, data_local)

    data = scipy.io.loadmat(data_local)
    X = data['X']
    y = data['y']
    for e in y:
        if e[0] == 10:
            e[0] = 0
    return X, y


def shuffle(X, y):
    order = []
    order.extend(range(X.shape[0]))
    random.shuffle(order)

    X_out = np.empty(X.shape)
    y_out = np.empty(y.shape)

    for i in order:
        X_out[i] = X[order[i]]
        y_out[i] = y[order[i]]
    return X_out, y_out
