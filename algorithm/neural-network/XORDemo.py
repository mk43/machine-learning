# coding: utf-8

import numpy as np

k = 2


def sigmoid(x):
    return 1 / (1 + np.power(np.e, -k * (x)))


def actication(data):
    return sigmoid(data)


def forward(W, data):
    z, a = [], []
    a.append(data)
    data = np.row_stack(([1], data))
    for w in W:
        z.append(np.matmul(w, data))
        a.append(actication(z[-1]))
        data = np.row_stack(([1], a[-1]))
    return z, a


def backward(y, W, z, a, learningrate):
    length = len(z) + 1
    Jtoz = k * (1 - y * (1 + np.power(np.e, -(k * z[-1])))) / np.power(np.e, -(k * z[-1]))
    # print("loss = %s" % (-y * np.log(a[-1]) - (1 - y) * np.log(1 - a[-1])))
    for layer in range(length - 1, 0, -1):
        i = layer - length
        if (i != -1):
            Jtoz = np.matmul(W[i + 1][:, 1:].T, Jtoz) * k * np.power(np.e, -(k * z[i])) / np.power(
                1 + np.power(np.e, -(k * z[i])), 2)
        W[i] = W[i] - learningrate * np.matmul(Jtoz, np.row_stack(([1], a[i - 1])).T)
    return W


def gradientDescent(X, y, shape, learningrate=0.001, trainingtimes=500):
    W, z, a = [], [], []
    W.append(np.asarray([[-1, 1, -1], [-1, -1, 1]]))
    W.append(np.asarray([[-1, 1, 1]]))
    # for layer in range(len(shape) - 1):
    #     row = shape[layer + 1]
    #     col = shape[layer] + 1
    #     W.append(np.random.normal(0, 1, row * col).reshape(row, col))
    for i in range(trainingtimes):
        for x, j in zip(X.T, range(len(X[0]))):
            z, a = forward(W, np.asarray([x]).T)
            W = backward(y[j], W, z, a, learningrate)
    return W


def train():
    np.set_printoptions(precision=4, suppress=True)
    x1 = np.asarray([0, 0, 1, 1])
    x2 = np.asarray([0, 1, 0, 1])
    X = np.row_stack((x1, x2))
    y = np.asarray([0, 1, 1, 0])
    shape = [2, 2, 1]
    Learning_Rate = 0.1
    Training_Times = 4000
    W = gradientDescent(X, y, shape, learningrate=Learning_Rate, trainingtimes=Training_Times)

    print(W)
    testData = np.row_stack((np.ones(shape=(1, 4)), X))
    for w in W:
        testData = np.matmul(w, testData)
        testData = np.row_stack((np.ones(shape=(1, 4)), actication(testData)))
    print(testData[1])


if __name__ == "__main__":
    train()

# output1:
# [array([[-8.3273,  5.5208,  5.4758],
#        [ 2.7417, -5.944 , -5.9745]]), array([[ 18.5644, -22.4426, -22.4217]])]
# [0.0005 1.     1.     0.0005]
# [array([[ 3.0903, -6.3961,  6.928 ],
#        [ 3.0355,  6.7901, -6.2563]]), array([[ 41.2259, -22.2455, -22.0939]])]
# [0.0024 1.     1.     0.0021]
# [array([[ 5.3893,  4.913 , -7.0756],
#        [ 6.2289, -1.3519, -4.7387]]), array([[  9.8004, -20.0023,  10.2014]])]
# [0.5    1.     0.4995 0.0002]

# output2:
# [array([[-2.8868,  5.6614, -5.9766],
#        [-2.9168, -5.9789,  5.6363]]), array([[-2.1866, 21.2065, 21.1815]])]
# [0.016  1.     1.     0.0142]
# [array([[-2.9942,  5.7925, -6.0901],
#        [-3.0228, -6.0924,  5.7687]]), array([[-3.6425, 22.3914, 22.3658]])]
# [0.0009 1.     1.     0.0008]