# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 6, 7, 8, 9, 10]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1]

train_X = np.asarray(x)
train_Y = np.asarray(y)

fig = plt.figure()
plt.xlim(-1, 12)
plt.ylim(-0.5, 1.5)
plt.scatter(train_X, train_Y)

s_X = np.linspace(-2, 12, 100)
s_Y = 1/(1 + np.power(np.e, -2*(s_X - 4)))

plt.plot(s_X, s_Y)
# plt.savefig("linear2LogisticreGressionGraphAddSigmoid3.png")
# plt.savefig("linear2LogisticreGressionGraphAddSigmoid2.png")
plt.show()
