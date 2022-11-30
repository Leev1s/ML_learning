#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   PureMath.py
@Time    :   2022/08/10 14:50:23
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
# import requests
# import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def prepare():
    train = pd.read_csv('mnist/mnist_train.csv')
    test = pd.read_csv('mnist/mnist_test.csv')
    train = np.array(train)
    test = np.array(test)
    m, n = train.shape
    i, j = test.shape
    np.random.shuffle(train)
    np.random.shuffle(test)
    return train,test

def imgSHOW(raw,showNO_):
    img = raw[showNO_,1:len(raw)]
    img = img.reshape(28,28)
    plt.figure(str(showNO_))
    plt.imshow(img,cmap = 'gray')
    plt.title(str(raw[showNO_,0]))
    plt.show()

def imgSHOWrand(raw):
    showNO_ = np.random.randint(0,raw.shape[0])
    img = raw[showNO_,1:len(raw)]
    img = img.reshape(28,28)
    plt.figure(str(showNO_))
    plt.imshow(img,cmap = 'gray')
    plt.title(str(raw[showNO_,0]))
    plt.show()

def initParams(train,test):
    x1 = train[:,1:len(train)]/255
    x2 = test[:,1:len(test)]/255
    a1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    a2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return a1,b1,a2,b2,x1.T,x2.T

def ReLU(x):
    return np.maximum(0,x)

def Softmax(x):
    A = np.exp(x) / sum(np.exp(x))
    return A

def forward(a1,b1,a2,b2,x1):
    y1 = a1.dot(x1) + b1
    Y1 = ReLU(y1)
    y2 = a2.dot(Y1) + b2
    Y2 = Softmax(y2)
    return y1,Y1,y2,Y2

def Floss(Y,lable):
    loss = np.zeros_like(Y)
    for it in range(60000):
        loss[lable[it],it] = 1
    loss = loss - Y
    return loss

def dReLU(x):
    return x > 0

def backward(x1,y1,Y1,a2,Y2,lable):
    dY2 = Floss(Y2,lable)
    da2 = 1 / m * dY2.dot(Y1.T)
    db2 = 1 / m * np.sum(dY2,axis = 1)
    db2 = db2.reshape(10,1)
    dy1 = dReLU(y1) * a2.T.dot(Y2)
    da1 = 1 / m * dy1.dot(x1.T)
    db1 = 1 / m * np.sum(dy1,axis = 1)
    db1 = db1.reshape(10,1)
    return da1,db1,da2,db2

def update_params(a1,b1,da1,db1,a2,b2,da2,db2,learnRate):
    a1 = a1 - learnRate * da1
    b1 = b1 - learnRate * db1 
    a2 = a2 - learnRate * da2
    b2 = b2 - learnRate * db2 
    return a1,b1,a2,b2

def prediction(Y2):
    result = np.argmax(Y2,0)
    print(result)
    print(trainLable)
    accuracy = np.sum(result == trainLable) / trainLable.size
    return accuracy


if __name__ == '__main__':
    looptimes = 500
    learnRate = 0.1
    # key = 'PDU14313Tq4eoHfkLLx5Hc1QNqYOvhJLTQ1BTUIqB'
    # server = "https://api2.pushdeer.com/message/push"
    train,test = prepare()
    #imgSHOWrand(train)
    #imgSHOWrand(test)
    a1,b1,a2,b2,x1,x2 = initParams(train,test)
    trainLable = train[:,0]
    testLable = test[:,0]
    m = trainLable.size
    for t in range(looptimes):
        y1,Y1,y2,Y2 = forward(a1,b1,a2,b2,x1)
        loss = Floss(Y2,trainLable)
        da1,db1,da2,db2 = backward(x1,y1,Y1,a2,Y2,trainLable)
        a1,b1,a2,b2 = update_params(a1,b1,da1,db1,a2,b2,da2,db2,learnRate)
        print(np.max(a1))
        if t % 10 == 0:
            print('LoopTime:',t)
            print(prediction(Y2))


    # try:
    #     requests.get(server+endpoint,params={
    #         'pushkey': key,
    #         'text': 'Finish',
    #     })
    # except:
    #     print('通知失败')
    # else:
    #     print('已通知')





    