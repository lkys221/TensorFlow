import tensorflow as tf
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

x = [1, 2]
y = [3, 4]
z = [5, 6]

print(tf.stack([x, y, z]).eval())
print(tf.stack([x, y, z], axis=1).eval())
print(tf.stack([x, y, z], axis=0).eval())


a = [[1., 2.],
     [3., 4.]]

print("---------------------------")
print(tf.reduce_sum(a, axis=0).eval())
print(tf.reduce_sum(a, axis=1).eval())