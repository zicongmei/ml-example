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
X_all=data['X']
y_all=data['y']

X_array=[]
y_array=[]
for i in range(len(X_all)):
    if y_all[i] == 1 or y_all[i] == 0:
        X_array.append(X_all[i])
        y_array.append(y_all[i])

X=np.empty([len(y_array), X.shape[1]])
y=np.empty([len(y_array),1])

for i in range(len(X_array)):
    if y_all[i] == 1 or y_all[i] == 0:
        X[i] = (X_array[i])
        y[i] = (y_array[i])



# print ('The first element of X is: ', X[0])
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))


def plot_X(X):
    m, n = X.shape

    fig, axes = plt.subplots(8,8, figsize=(8,8))
    fig.tight_layout(pad=0.1)

    for i,ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image
        X_random_reshaped = X[random_index].reshape((20,20)).T
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Display the label above the image
        ax.set_title(y[random_index,0])
        ax.set_axis_off()
    plt.show()

# plot_X(X)

model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    #specify input size
        ### START CODE HERE ### 
        Dense(25, activation="sigmoid", name="layer1"),
        Dense(15, activation="sigmoid", name="layer2"),
        Dense(1, activation="sigmoid", name="layer3"),
        
        ### END CODE HERE ### 
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