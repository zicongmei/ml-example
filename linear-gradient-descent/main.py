import math, copy
import numpy as np

print("program started")

x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

def fwb(x, w, b):
  return w * x + b
def compute_cost(x, y, w, b):
  m = len(x)
  cost = 0.0
  for i in range(m):
    cost += (fwb(x[i], w, b) - y[i]) ** 2
  return cost / 2 / m


def compute_gradient(x, y, w, b):
  m = len(x)

  djdw = 0.0
  djdb = 0.0

  for i in range (m):
    djdw += (fwb(x[i], w, b) - y[i]) * x[i]
    djdb += fwb(x[i], w, b) - y[i]

  djdw /= m
  djdb /= m

  return djdw, djdb

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
  J_history = []
  p_history = []
  b = b_in
  w = w_in

  for i in range(num_iters):
    # Calculate the gradient and update the parameters using gradient_function
    dj_dw, dj_db = gradient_function(x, y, w , b)

    # Update Parameters using equation (3) above
    b = b - alpha * dj_db
    w = w - alpha * dj_dw

    if i<100000:      # prevent resource exhaustion
      J_history.append( cost_function(x, y, w , b))
      p_history.append([w,b])
      # Print cost every at intervals 10 times or as many iterations if < 10
    if i% math.ceil(num_iters/10) == 0:
      print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
            f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
            f"w: {w: 0.3e}, b:{b: 0.5e}")

  return w, b, J_history, p_history #return w and J,w history for graphing

w_init = 0
b_init = 0
# some gradient descent settings
iterations = 1000
tmp_alpha = 0.1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")