"""
Linear regression model that predicts what cities can give higher profits 
for a business based on available data on the city populations.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math

# load the dataset
x_train = np.load('x.npy')
y_train = np.load('y.npy')

# Cost function for linear regression
def compute_cost(x, y, w, b): 

    # number of training examples
    m = x.shape[0]
    
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i])**2
        total_cost += cost

    total_cost /= 2*m

    return total_cost

# Gradient computation for gradient_descent 
def compute_gradient(x, y, w, b): 

    # Number of training examples
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_db_i = f_wb - y[i]
        dj_dw_i = dj_db_i * x[i]
        
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw /= m
    dj_db /= m
        
    return dj_dw, dj_db

# Gradient descent computation
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):     
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing

initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.savefig("profitVsPopulation.png")

pplt = int(input("What is the population of the city? \n- "))
pplt2 = pplt / 10000

predict1 = pplt2 * w + b
print(f'For population = {pplt}, we predict a profit of $%.2f' % (predict1*10000))
print("\nThe plot of the model is saved as profitVsPopulation.png!")