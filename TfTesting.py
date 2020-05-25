from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
sess=tf.Session()

# a rank 0 tensor; a scalar with shape [],
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)# assume it is a 32 bit float

total= a + b 

print(a)
print(b)
print(total)
print('This is working')

writer = tf.summary.FileWriter('/home/dan/python')
writer.add_graph(tf.get_default_graph())
