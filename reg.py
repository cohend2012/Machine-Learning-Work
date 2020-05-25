from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt


#%cofig InLineBackend.figure_format ='svg'

learning_rate = 0.01
epochs= 200

n_samples = 30

train_x = np.linspace(0,20,n_samples)
train_y = 3*train_x+4*np.random.randn(n_samples)
plt.plot(train_x,train_y,'o')
plt.show()

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(np.random.randn(),name='weights')
B = tf.Variable(np.random.randn(),name='bias')

pred = X * W + B

#graph

#pred = tf.add(tf.multiply(X,W),B)


#cost fun

cost = tf.reduce_sum((pred - Y)**2)/(2*n_samples)

#optimizer

optimizer= tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as sesh:

	sesh.run(init)
	
	for epoch in range(epochs):
		for x,y in zip(train_x,train_y):
			sesh.run(optimizer, feed_dict={X:x,Y:y})
			
		if not epoch % 20:
			c=sesh.run(cost,feed_dict={X: train_x,Y:train_y})
			w=sesh.run(W)
			b=sesh.run(B)
			print(f'epoch: {epoch:04d} c={c:.4f} w={w:.4f} b={b:.4f}')
			
			
			
	weight=sesh.run(W)	
	bias=sesh.run(B)	
	
	plt.plot(train_x,train_y,'o')
	plt.plot(train_x,weight*train_x+bias)
	plt.show()
