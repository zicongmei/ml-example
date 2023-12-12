#!/usr/bin/python
import copy, math
import numpy as np

print("program started")

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

def predict_single_loop(x, w, b):
  """
  single predict using linear regression

  Args:
    x (ndarray): Shape (n,) example with multiple features
    w (ndarray): Shape (n,) model parameters
    b (scalar):  model parameter

  Returns:
    p (scalar):  prediction
  """
  n = x.shape[0]
  p = 0
  for i in range(n):
    p_i = x[i] * w[i]
    p = p + p_i
  p = p + b
  return p


def predict(x, w, b):
  """
  single predict using linear regression
  Args:
    x (ndarray): Shape (n,) example with multiple features
    w (ndarray): Shape (n,) model parameters
    b (scalar):             model parameter

  Returns:
    p (scalar):  prediction
  """
  p = np.dot(x, w) + b
  return p


def compute_cost(X, y, w, b):
  """
  compute cost
  Args:
    X (ndarray (m,n)): Data, m examples with n features
    y (ndarray (m,)) : target values
    w (ndarray (n,)) : model parameters
    b (scalar)       : model parameter

  Returns:
    cost (scalar): cost
  """
  m = X.shape[0]
  cost = 0.0
  for i in range(m):
    f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
    cost = cost + (f_wb_i - y[i])**2       #scalar
  cost = cost / (2 * m)                      #scalar
  return cost

def my_cost(X, y, w, b):
  """
  compute cost
  Args:
    X (ndarray (m,n)): Data, m examples with n features
    y (ndarray (m,)) : target values
    w (ndarray (n,)) : model parameters
    b (scalar)       : model parameter

  Returns:
    cost (scalar): cost
  """
  diff = np.dot(X, w) + b - y
  cost = np.dot(diff, diff) / 2 / len(y)
  return cost

# cost = compute_cost(X_train, y_train, w_init, b_init)
# print(f'Cost at optimal w : {cost}')
# cost = my_cost(X_train, y_train, w_init, b_init)
# print(f'Cost at optimal w : {cost}')

def compute_gradient(X, y, w, b):
  """
  Computes the gradient for linear regression
  Args:
    X (ndarray (m,n)): Data, m examples with n features
    y (ndarray (m,)) : target values
    w (ndarray (n,)) : model parameters
    b (scalar)       : model parameter

  Returns:
    dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
    dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
  """
  m,n = X.shape           #(number of examples, number of features)
  dj_dw = np.zeros((n,))
  dj_db = 0.

  for i in range(m):
    err = (np.dot(X[i], w) + b) - y[i]
    for j in range(n):
      dj_dw[j] = dj_dw[j] + err * X[i, j]
    dj_db = dj_db + err
  dj_dw = dj_dw / m
  dj_db = dj_db / m

  return dj_db, dj_dw

def my_gradient(X, y, w, b):
  dj_dw = np.dot(predict(X,w,b)-y,X) / len(y)
  dj_db = sum(predict(X,w,b)-y)/len(y)
  return dj_db, dj_dw

# tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
# print(f'dj_db at initial w,b: {tmp_dj_db}')
# print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
#
# tmp_dj_db, tmp_dj_dw = my_gradient(X_train, y_train, w_init, b_init)
# print(f'dj_db at initial w,b: {tmp_dj_db}')
# print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
  """
  Performs batch gradient descent to learn w and b. Updates w and b by taking
  num_iters gradient steps with learning rate alpha

  Args:
    X (ndarray (m,n))   : Data, m examples with n features
    y (ndarray (m,))    : target values
    w_in (ndarray (n,)) : initial model parameters
    b_in (scalar)       : initial model parameter
    cost_function       : function to compute cost
    gradient_function   : function to compute the gradient
    alpha (float)       : Learning rate
    num_iters (int)     : number of iterations to run gradient descent

  Returns:
    w (ndarray (n,)) : Updated values of parameters
    b (scalar)       : Updated value of parameter
    """

  # An array to store cost J and w's at each iteration primarily for graphing later
  J_history = []
  w = copy.deepcopy(w_in)  #avoid modifying global w within function
  b = b_in

  for i in range (num_iters):
    dj_db, dj_dw = my_gradient(X, y, w, b)
    w -= alpha * dj_db
    b -= alpha * dj_db

    J_history.append( cost_function(X, y, w, b))

    # Print cost every at intervals 10 times or as many iterations if < 10
    if i% math.ceil(num_iters / 10) == 0:
      print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

  return w, b, J_history #return final w,b and J history for graphing

# initial_w = np.zeros_like(w_init)
initial_w = np.array([460.0, 232, 178, 300])
print(np.shape(initial_w))
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 0.0001
# run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                            compute_cost, compute_gradient,
                                            alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
  print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")