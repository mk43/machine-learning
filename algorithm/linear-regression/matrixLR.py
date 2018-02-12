# coding: utf-8

import numpy as np

X1 = np.asarray([2104, 1416, 1534, 852]).reshape(4, 1)
X2 = np.asarray([5, 3, 3, 2]).reshape(4, 1)
X3 = np.asarray([1, 2, 2, 1]).reshape(4, 1)

X = np.mat(np.column_stack((X1, X2, X3, np.ones(shape=(4, 1)))))
noise = np.random.normal(0, 0.1, X1.shape)
Y = np.mat(2.5 * X1 - X2 + 2 * X3 + 4 + noise)
YTwin = np.mat(2.5 * X1 - X2 + 2 * X3 + 4)

W = (X.T * X).I * X.T * Y
WTWin = (X.T * X).I * X.T * YTwin
print(W, "\n", WTWin)
