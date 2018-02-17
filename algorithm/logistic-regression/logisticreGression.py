# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

x = [1, 2, 3, 4, 6, 7, 8, 9, 10]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1]

train_X = np.asarray(np.row_stack((np.ones(shape=(1, len(x))), x)), dtype=np.float64)
train_Y = np.asarray(y, dtype=np.float64)
train_W = np.asarray([-1, -1], dtype=np.float64).reshape(1, 2)


def sigmoid(X):
    return 1 / (1 + np.power(np.e, -(X)))


def lossfunc(X, Y, W):
    n = len(Y)
    return (-1 / n) * np.sum(Y * np.log(sigmoid(np.matmul(W, X))) + (1 - Y) * np.log((1 - sigmoid(np.matmul(W, X)))))


Training_Times = 100000
Learning_Rate = 0.3

loss_Trace = []
w_Trace = []
b_Trace = []


def gradientDescent(X, Y, W, learningrate=0.001, trainingtimes=500):
    n = len(Y)
    for i in range(trainingtimes):
        W = W - (learningrate / n) * np.sum((sigmoid(np.matmul(W, X)) - Y) * X, axis=1)

        # for GIF
        if 0 == i % 1000 or (100 > i and 0 == i % 2):
            b_Trace.append(W[0, 0])
            w_Trace.append(W[0, 1])
            loss_Trace.append(lossfunc(X, Y, W))
    return W


final_W = gradientDescent(train_X, train_Y, train_W, learningrate=Learning_Rate, trainingtimes=Training_Times)

print("Final Weight:", final_W)
print("Weight details trace: ", np.asarray([b_Trace, w_Trace]))
print("Loss details trace: ", loss_Trace)

fig, ax = plt.subplots()
ax.scatter(np.asarray(x), np.asarray(y))
ax.set_title(r'$Fitting\ line$')


def update(i):
    try:
        ax.lines.pop(0)
    except Exception:
        pass
    plot_X = np.linspace(-1, 12, 100)
    W = np.asarray([b_Trace[i], w_Trace[i]]).reshape(1, 2)
    X = np.row_stack((np.ones(shape=(1, len(plot_X))), plot_X))
    plot_Y = sigmoid(np.matmul(W, X))
    line = ax.plot(plot_X, plot_Y[0], 'r-', lw=1)
    ax.set_xlabel(r"$Cost\ %.6s$" % loss_Trace[i])
    return line


ani = animation.FuncAnimation(fig, update, frames=len(w_Trace), interval=100)
ani.save('logisticregression.gif', writer='imagemagick')

plt.show()
