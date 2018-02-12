# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

N = 200

X = np.linspace(0, 10, N * 2)
noise = np.random.normal(0, 0.5, X.shape)
Y = X * 0.5 + 3 + noise


def calcLoss(train_X, train_Y, W, b):
    return np.sum(np.square(train_Y - (train_X * W + b)))

def gradientDescent(train_X, train_Y, W, b, learningrate=0.001, trainingtimes=500):
    global loss
    global W_trace
    global b_trace
    size = train_Y.size
    for _ in range(trainingtimes):
        prediction = W * train_X + b
        tempW = W + learningrate * np.sum(train_X * (train_Y - prediction)) / size
        tempb = b + learningrate * np.sum(train_Y - prediction) / size
        W = tempW
        b = tempb
        loss.append(calcLoss(train_X, train_Y, W, b))
        W_trace.append(W)
        b_trace.append(b)


Training_Times = 100
Learning_Rate = 0.002

loss = []
W_trace = [-1]
b_trace = [1]
gradientDescent(X, Y, W_trace[0], b_trace[0], learningrate=Learning_Rate, trainingtimes=Training_Times)
print(W_trace[-1], b_trace[-1])

fig = plt.figure()
plt.title(r'$loss\ function\ change\ tendency$')
plt.xlabel(r'$learning\ times$')
plt.ylabel(r'$loss\ value$')
plt.plot(np.linspace(1, Training_Times, Training_Times), loss)
plt.savefig("gradientDescentLR.png")
plt.show()
