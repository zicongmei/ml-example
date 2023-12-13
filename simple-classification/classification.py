import numpy as np
import matplotlib.pyplot as plt
# from utils import *
import copy
import math


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


def fwb(x, w, b):
        inside = np.dot(w,x) + b
        result = sigmoid(inside)
        return result

def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    
    total_cost = 0
    
    for i in range(m):
        xi=X[i]
        loss = -1.0 * y[i] * np.log(fwb(xi,w,b)) -  (1.0 - y[i]) * np.log(1.0 - fwb(xi,w,b))
    
        total_cost += loss
        
    total_cost /= m    
    return total_cost


m, n = X_train.shape

# # Compute and display cost with w and b initialized to zeros
# initial_w = np.zeros(n)
# initial_b = 0.
# cost = compute_cost(X_train, y_train, initial_w, initial_b)
# print('Cost at initial w and b (zeros): {:.3f}'.format(cost))


def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    
    
    for i in range(m):
        xi=X[i]
        
        fwb_result = fwb(xi, w, b)
        dj_db += fwb_result - y[i]
        
        for j in range(n):
            dj_dw[j] += (fwb_result - y[i]) * xi[j]
            
    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw


# initial_w = np.zeros(n)
# initial_b = 0.

# dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
# print(f'dj_db at initial w and b (zeros):{dj_db}' )
# print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant
      
    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing


np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    # Loop over each example
    for i in range(m): 
        xi = X[i]
        fxi=sigmoid(np.dot(w, xi) + b)
        

        if fxi > 0.5:
            p[i] = 1
        else:
            p[i] = 0
        
    ### END CODE HERE ### 
    return p