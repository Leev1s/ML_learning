#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   wowowow copy.py
@Time    :   2022/11/30 10:03:45
@Author  :   Lev1s
@Version :   1.0
@Contact :   Lev1sStudio.cn@gmail.com
@PW      :   http://Lev1s.cn
@Github  :   https://github.com/o0Lev1s0o

'''
print('''
    __             ___        _____ __            ___     
   / /   ___ _   _<  /____   / ___// /___  ______/ (_)___ 
  / /   / _ \ | / / / ___/   \__ \/ __/ / / / __  / / __ \\
 / /___/  __/ |/ / (__  )   ___/ / /_/ /_/ / /_/ / / /_/ /
/_____/\___/|___/_/____/   /____/\__/\__,_/\__,_/_/\____/
''')
# here put the import lib
import pandas as pd
import numpy as np
print('pandas:',pd.__version__)
print('numpy:',np.__version__)

data = np.array(pd.read_csv('data.csv'))
X = data[:,:-1]
label = data[:,-1].astype(int)

n = label.size ##样本个数
p_n = X.shape[1] ##参数个数

y = np.zeros((n,3))
for i in range(n):
     y[i,label[i]] = 1
y=y.T

x = ((X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))).T

def init_params():
    W2 = np.random.rand(3, p_n) - 0.5
    b2 = np.random.rand(3, 1) - 0.5
    return W2, b2

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W2, b2, X):
    Z2 = W2.dot(X) + b2
    A2 = softmax(Z2)
    return Z2, A2

def backward_prop(A2, X, Y):
    dZ2 = A2 - Y
    dW2 = 1 / n * dZ2.dot(X.T)
    db2 = 1 / n * np.sum(dZ2)
    return dW2, db2

def update_params(W2, b2, dW2, db2, alpha):  
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions):
    print(predictions)
    return np.sum(predictions == label) / label.size

def gradient_descent(X, Y, alpha, iterations):
    W2, b2 = init_params()
    for i in range(iterations):
        Z2, A2 = forward_prop(W2, b2, X)
        dW2, db2 = backward_prop(A2, X, Y)
        W2, b2 = update_params(W2, b2, dW2, db2, alpha)
        if i % (iterations/10) == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions))
    return W2, b2

W2, b2 = gradient_descent(x, y, 0.01, 100000)

def make_predictions(X, W2, b2):
    _, A2 = forward_prop(W2, b2, X)
    return A2

print('----------------------------------')
test = np.array([[7.6,3.9,6.6,2.3],[5.2,3.9,1.4,0.1]]).T
print(make_predictions(test, W2, b2))
print(np.argmax(make_predictions(test, W2, b2),0))