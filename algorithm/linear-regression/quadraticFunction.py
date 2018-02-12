# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

N = 200

X = np.linspace(0, 10, N * 2)
noise = np.random.normal(0, 0.1, X.shape)
Y = X * 0.5 + 3 + noise

W = 1
b = np.linspace(-5, 6, N)

h = []
for i in b:
    h.append(np.sum(np.square(Y - (X * W + i))))

plt.plot(b, h)

# plt.savefig("quadraticFunction.png")

plt.show()