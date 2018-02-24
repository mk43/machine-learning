# coding: utf-8

import numpy as np
import tensorflow as tf


def sigmoid(x):
    return 1 / (1 + np.power(np.e, -2 * (x)))


def add_layer(inputs, in_size, out_size, activation_function=None, ):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


if __name__ == "__main__":
    x1 = np.asarray([0, 0, 1, 1])
    x2 = np.asarray([0, 1, 0, 1])
    X = np.row_stack((x1, x2))
    y = np.asarray([0, 1, 1, 0]).reshape(1, 4)
    data_X = tf.placeholder(tf.float32, [None, 2])
    data_y = tf.placeholder(tf.float32, [None, 1])


    layer_one = add_layer(data_X, 2, 2, activation_function=sigmoid)
    prediction = add_layer(layer_one, 2, 1, activation_function=sigmoid)
    # layer_one = add_layer(data_X, 2, 2, activation_function=tf.nn.sigmoid)
    # prediction = add_layer(layer_one, 2, 1, activation_function=tf.nn.sigmoid)

    loss = tf.reduce_mean(tf.reduce_sum(- data_y * tf.log(prediction) - (1 - data_y) * tf.log(1 - prediction)))
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(4000):
            sess.run(train, feed_dict={data_X: X.T, data_y: y.T})
        print(sess.run(prediction, feed_dict={data_X: X.T, data_y: y.T}))

# output:
# [[0.00200064]
#  [0.9985947 ]
#  [0.9985983 ]
#  [0.00144795]]
# --------------
# [[0.01765717]
#  [0.98598236]
#  [0.98598194]
#  [0.0207849 ]]
# --------------
# [[0.00104381]
#  [0.9991435 ]
#  [0.49951136]
#  [0.5003463 ]]
