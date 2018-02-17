# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# x = [1, 2, 3, 4, 6, 7, 8, 9, 10]
# y = [0, 0, 0, 0, 1, 1, 1, 1, 1]

# train_X = np.asarray(x)
# train_Y = np.asarray(y)

fig = plt.figure()
plt.xlim(-1, 12)
plt.ylim(1, 5)
# plt.scatter(train_X, train_Y)

# s_X = np.linspace(-2, 12, 100)
# s_Y = 1/(1 + np.power(np.e, -6*(s_X - 5)))

x = np.asarray([1, 1, 2, 1.3, 2.4, 4, 3.4, 4.4, 6, 4, 5, 6.2, 7, 7.7])
y = np.asarray([2, 3, 2.4, 3, 2.4, 4, 3, 2, 2.5, 4.4, 3.5, 3, 4, 2.5])

for _x, _y in zip(x, y):
    if _x + _y > 7:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'b^')
    else:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'ro')

# plt.plot(s_X, s_Y)
# plt.savefig("linear2LogisticreGressionGraphAddScatter.png")
# plt.savefig("linear2LogisticreGressionGraphAddSigmoid.png")
# plt.savefig("linear2LogisticreGressionGraph.png")
plt.show()
