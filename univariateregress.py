from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = pd.read_csv("Salary_Data.csv")

X = np.array(data['YearsExperience']).reshape(-1,1)
Y = np.array(data['Salary']).reshape(-1,1)

learning_rate = 0.01
iteration_num = 1000

x = X
y = Y
learning_rate =  0.01

#init starting thetas to zero
starting_theta_0 = 0
starting_theta_1 = 0

#OLS
ne_theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)


def derivative_J_theta(x, y, theta_0, theta_1):

  delta_J_theta0 = 0
  delta_J_theta1 = 0

  for i in range(len(x)):
    delta_J_theta0 +=  (((theta_1 * x[i]) + theta_0) - y[i])
    delta_J_theta1 +=  (1/x.shape[0]) * (((theta_1 * x[i]) + theta_0) - y[i]) * x[i]
  
  temp0 = theta_0 - (learning_rate * ((1/x.shape[0]) * delta_J_theta0) )
  temp1 = theta_1 - (learning_rate * ((1/x.shape[0]) * delta_J_theta1) )
  
  return temp0, temp1

#Gradient Descent
def gradient_descent(x, y, learning_rate, starting_theta_0, starting_theta_1, iteration_num):
  store_theta_0 = np.empty([iteration_num])
  store_theta_1 = np.empty([iteration_num])
  # store_j_theta = []

  theta_0 = starting_theta_0
  theta_1 = starting_theta_1

  for i in range(iteration_num):
    theta_0, theta_1 = derivative_J_theta(x, y, theta_0, theta_1)
    store_theta_0[i] = theta_0
    store_theta_1[i] = theta_1
    store_j_theta = ((1/2*X.shape[0]) * ( ((theta_1 * X) + theta_0) - Y)**2)
    # store_j_theta.append((1/2*X.shape[0]) * ( ((theta_1 * X) + theta_0) - Y)**2)


  return theta_0, theta_1, store_theta_0, store_theta_1, store_j_theta

theta_0, theta_1, store_theta_0, store_theta_1, store_j_theta = gradient_descent(x, y, learning_rate, starting_theta_0, starting_theta_1, iteration_num)

print("m : %f" %theta_0[0])
print("b : %f" %theta_1[0])


plt.scatter(X[:, 0], Y[:, 0])
plt.plot(X,(theta_1 * X) + theta_0, c='blue')
plt.plot(X, X.dot(ne_theta), c = 'red')
plt.title('r=%f'%ne_theta[0,0])
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $")
plt.show()


