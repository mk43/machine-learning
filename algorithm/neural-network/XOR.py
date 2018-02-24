# coding: utf-8

import numpy as np

x1 = np.asarray([0, 0, 1, 1])
x2 = np.asarray([0, 1, 0, 1])
X = np.row_stack((np.ones(shape=(1, 4)), x1, x2))
print("X:\n%s" % X)
y = np.asarray([0, 1, 1, 0])
W1 = np.asarray([[-1, 2, -2],
                 [-1, -2, 2]])
W2 = np.asarray([-1, 2, 2])


def sigmoid(input):
    return 1 / (1 + np.power(np.e, -10 * (input)))


np.set_printoptions(precision=6, suppress=True)
z1 = np.matmul(W1, X)
print("W1*X = z1:\n%s" % z1)
a1 = np.row_stack((np.ones(shape=(1, 4)), sigmoid(z1)))
print("sigmoid(z1) = a1:\n%s" % a1)
z2 = np.matmul(W2, a1)
print("W2*a1 = z2:\n%s" % z2)
a2 = sigmoid(z2)
print("------------------------")
print("prediction: %s" % a2)
print("target: %s" % y)
print("------------------------")

# output:
# X:
# [[1. 1. 1. 1.]
#  [0. 0. 1. 1.]
#  [0. 1. 0. 1.]]
# W1*X = z1:
# [[-1. -3.  1. -1.]
#  [-1.  1. -3. -1.]]
# sigmoid(z1) = a1:
# [[1.       1.       1.       1.      ]
#  [0.000045 0.       0.999955 0.000045]
#  [0.000045 0.999955 0.       0.000045]]
# W2*a1 = z2:
# [-0.999818  0.999909  0.999909 -0.999818]
# ------------------------
# prediction: [0.000045 0.999955 0.999955 0.000045]
# target: [0 1 1 0]
# ------------------------
