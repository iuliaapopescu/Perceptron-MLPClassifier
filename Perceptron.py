
import matplotlib.pyplot as plt 

from sklearn.linear_model import Perceptron
from sklearn import preprocessing

import numpy as np

def plot3d_data(X, y):
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2],'b');
    ax.scatter3D(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],'r'); 
    plt.show()
    
def plot3d_data_and_decision_function(X, y, W, b): 
    ax = plt.axes(projection='3d')
    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))
    # calculate corresponding z
    # [x, y, z] * [coef1, coef2, coef3] + b = 0
    zz = (-W[0] * xx - W[1] * yy - b) / W[2]
    ax.plot_surface(xx, yy, zz, alpha=0.5) 
    ax.scatter3D(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2],'b');
    ax.scatter3D(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],'r'); 
    plt.show()
    
X_train = np.loadtxt('data/3d-points/x_train.txt')
y_train = np.loadtxt('data/3d-points/y_train.txt', 'int') 

plot3d_data(X_train, y_train)
X_test = np.loadtxt('data/3d-points/x_test.txt')
y_test = np.loadtxt('data/3d-points/y_test.txt', 'int') 


scaler = preprocessing.StandardScaler()

scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

perceptron_model = Perceptron(eta0=0.1, tol=1e-5)
perceptron_model.fit(X_train_sc, y_train)
accuracy = perceptron_model.score(X_test_sc, y_test)

print(accuracy)

no_iter = perceptron_model.n_iter_
bias = perceptron_model.intercept_
Weights = perceptron_model.coef_.reshape(3, 1)

print(no_iter)
print(bias)
print(Weights)

plot3d_data_and_decision_function(X_train_sc,
                                  y_train,
                                  Weights,
                                  bias)



