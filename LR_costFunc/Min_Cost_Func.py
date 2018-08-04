import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypo = W * X
cost = tf.reduce_mean(tf.square(hypo = Y))

learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)