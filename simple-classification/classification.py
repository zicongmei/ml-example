import numpy as np
import matplotlib.pyplot as plt
# from utils import *
import copy
import math
import python_utils


print("starting")

# load dataset
data_source="https://raw.githubusercontent.com/kaleko/CourseraML/master/ex2/data/ex2data1.txt"
loaded_data = np.genfromtxt(data_source, delimiter=',')
m,n=loaded_data.shape
X_train=loaded_data[:,0:2]
y_train=loaded_data[:,2]

print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

plt.scatter(X_train[:,0], X_train[:,1],c=y_train)
plt.show()

def sigmoid(z):
  """
  Compute the sigmoid of z

  Args:
      z (ndarray): A scalar, numpy array of any size.

  Returns:
      g (ndarray): sigmoid(z), with the same shape as z

  """

  up = np.ones(np.shape(z))
  down = (1 + np.exp(np.multiply(-1.0, z)))
  g = np.divide ( up,down )
  return g

# value = np.array([-1, 0, 1, 2])
# print (f"sigmoid({value}) = {sigmoid(value)}")

