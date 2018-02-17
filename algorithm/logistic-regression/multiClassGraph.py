# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(10, 4, 100)
y = np.random.normal(10, 4, 100)

plt.figure()
for _x, _y in zip(x, y):
    if _x ** 2 + _y ** 2 < 200:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'b^')
    elif - _x + _y > 1:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'ro')
    else:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'gs')
# plt.savefig("multiClass3ToAll.png")
plt.show()

plt.figure()
for _x, _y in zip(x, y):
    if _x ** 2 + _y ** 2 < 200:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'b^')
    elif - _x + _y > 1:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'ro')
    else:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'rs')
# plt.savefig("multiClass3To1.png")
plt.show()

plt.figure()
for _x, _y in zip(x, y):
    if _x ** 2 + _y ** 2 < 200:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'g^')
    elif - _x + _y > 1:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'ro')
    else:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'gs')
# plt.savefig("multiClass3To2.png")
plt.show()

plt.figure()
for _x, _y in zip(x, y):
    if _x ** 2 + _y ** 2 < 200:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'b^')
    elif - _x + _y > 1:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'bo')
    else:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'gs')
# plt.savefig("multiClass3To3.png")
plt.show()

plt.figure()
x = np.random.normal(0, 2, 200)
y = np.random.normal(0, 2, 200)
for _x, _y in zip(x, y):
    if _x ** 2 + _y ** 2 < 8:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'b^')
    else:
        plt.plot(np.asarray([_x]), np.asarray([_y]), 'ro')

boderparameter = plt.gca()
boderparameter.spines['right'].set_position(('data', 0))
boderparameter.spines['top'].set_position(('data', 0))
plt.savefig("circleGraph.png")
plt.show()
