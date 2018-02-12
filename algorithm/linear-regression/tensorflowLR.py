# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

N = 1000
train_X1 = np.linspace(1, 10, N).reshape(N, 1)
train_X2 = np.linspace(1, 10, N).reshape(N, 1)
train_X3 = np.linspace(1, 10, N).reshape(N, 1)
train_X4 = np.linspace(1, 10, N).reshape(N, 1)

# train_X = np.column_stack((train_X1, np.ones(shape=(N, 1))))
train_X = np.column_stack((train_X1, train_X2, train_X3, train_X4, np.ones(shape=(N, 1))))

noise = np.random.normal(0, 0.5, train_X1.shape)
# train_Y = 3 * train_X1 + 4
train_Y = train_X1 + train_X2 + train_X3 + train_X4 + 4 + noise

length = len(train_X[0])

X = tf.placeholder(tf.float32, [None, length], name="X")
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

W = tf.Variable(np.random.random(size=length).reshape(length, 1), dtype=tf.float32, name="weight")

activation = tf.matmul(X, W)
learning_rate = 0.006

loss = tf.reduce_mean(tf.reduce_sum(tf.pow(activation - Y, 2), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

training_epochs = 2000
display_step = 100

loss_trace = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})
        temp_loss = sess.run(loss, feed_dict={X: train_X, Y: train_Y})
        loss_trace.append(temp_loss)
        if 1 == epoch % display_step:
            print('epoch: %4s'%epoch, '\tloss: %s'%temp_loss)
    print("\nOptimization Finished!")
    print("\nloss = ", loss_trace[-1], "\nWeight =\n", sess.run(W, feed_dict={X: train_X, Y: train_Y}))


plt.plot(np.linspace(0, 100, 100), loss_trace[:100])
# plt.savefig("tensorflowLR.png")
plt.show()

# output:
# epoch:    1 	loss: 118.413925
# epoch:  101 	loss: 1.4500043
# epoch:  201 	loss: 1.0270562
# epoch:  301 	loss: 0.75373846
# epoch:  401 	loss: 0.5771168
# epoch:  501 	loss: 0.46298113
# epoch:  601 	loss: 0.38922414
# epoch:  701 	loss: 0.34156123
# epoch:  801 	loss: 0.31076077
# epoch:  901 	loss: 0.29085675
# epoch: 1001 	loss: 0.27799463
# epoch: 1101 	loss: 0.26968285
# epoch: 1201 	loss: 0.2643118
# epoch: 1301 	loss: 0.26084095
# epoch: 1401 	loss: 0.2585978
# epoch: 1501 	loss: 0.25714833
# epoch: 1601 	loss: 0.25621164
# epoch: 1701 	loss: 0.2556064
# epoch: 1801 	loss: 0.2552152
# epoch: 1901 	loss: 0.2549625
# Optimization Finished!
# loss =  0.25480175
# Weight =
#  [[1.0982682 ]
#  [0.9760315 ]
#  [1.0619627 ]
#  [0.87049955]
#  [3.9700394 ]]
