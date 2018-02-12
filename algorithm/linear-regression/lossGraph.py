# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel(r'$Weight$')
ax.set_ylabel(r'$bias$')
ax.set_zlabel(r'$loss\ value$')
N = 200

X = np.linspace(0, 10, N*2)
noise = np.random.normal(0, 0.1, X.shape)
Y = X * 0.5 + 3 + noise

W = np.linspace(-10, 10, N)
b = np.linspace(-2, 8, N)

p = 0.6
plt.xlim(W.min() * p, W.max() * p)
plt.ylim(b.min() * p, b.max() * p)
ax.set_zlim(0, 1000)

h = []
for i in W:
    for j in b:
        h.append(np.sum(np.square(Y - (X * i + j))))

print(np.min(h))
h = np.asarray(h).reshape(N, N)
W, b = np.meshgrid(W, b)

ax.scatter(W, b, h, c='b')
# plt.savefig("lossgraph.png")

plt.show()
