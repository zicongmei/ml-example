import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import logging
import scipy.io
import urllib.request
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

data_url='https://raw.githubusercontent.com/kaleko/CourseraML/master/ex3/data/ex3data1.mat'
data_local = "/tmp/ex3data1.mat"
urllib.request.urlretrieve(data_url, data_local)

data = scipy.io.loadmat(data_local)
X=data['X']
y=data['y']

# print ('The first element of X is: ', X[0])
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))

def plot_X(X,y):
    m, n = X.shape
    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1)
    for i,ax in enumerate(axes.flat):
        X_random_reshaped = X[i].reshape((20,20)).T
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        # Display the label above the image
        ax.set_title(round(y[i,0],2))
        ax.set_axis_off()
    plt.show()

def plot_random_X(X,y):
    size = 64
    X_out = np.empty([size,X.shape[1]])
    y_out = np.empty([size,y.shape[1]])
    m,n=X.shape
    for i in range(size):
        # Select random indices
        random_index = np.random.randint(m)
        X_out[i] = X[random_index]
        y_out[i] = y[random_index]
    plot_X(X_out,y_out)    

plot_random_X(X,y)

model = Sequential(
    [               
        tf.keras.Input(shape=(X.shape[1],)),  
        Dense(25, activation="relu", name="layer1"),
        Dense(15, activation="relu", name="layer2"),
        Dense(10, activation="linear", name="layer3"),
    ], name = "my_model" 
)                            
model.summary()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)
model.fit(
    X,y,
    epochs=20
)

prediction = model.predict(X)
perdition_p = tf.nn.softmax(prediction)

plot_random_X(X,perdition_p)
