# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def draw():
    X = np.linspace(0, 10, 50)
    noise = np.random.normal(0, 0.5, X.shape)
    Y = X * 0.5 + 3 + noise

    plt.scatter(X, Y)
    plt.savefig("scattergraph.png")
    plt.show()

if __name__ == "__main__":
    draw()