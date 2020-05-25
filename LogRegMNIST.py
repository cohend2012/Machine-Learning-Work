from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt


#%cofig InLineBackend.figure_format ='svg'

(x_train,y_train), (x_test,y_test)=tf.keras.datasets.mnist.load_data()

fig, axes = plt.subplots(1, 4, figsize=(7,3))
for img, label, ax in zip(x_train[:4], y_train[:4], axes):
	ax.set_title(label)
	ax.imshow(img)
	ax.axis('off')
plt.show()

print(f'train images: {x_train.shape}')
print(f'train labels: {y_train.shape}')
print(f' test images: {x_test.shape}')
print(f' test images: {y_test.shape}')

#Preprocessing

x_train= x_train.reshape(60000,784)/255
x_test= x_test.reshape(10000,784)/255


	
# hyper parameters

learning_rate = 0.01
epochs = 100
batch_size = 100
batches =int(x_train.shape[0]/batch_size) 

# inputs
# X is our "flattened /normilizd images"
# Y is our "one hot" lables
#DATA
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#WHAT WILL LEARN
W=tf.Variable(0.1*np.random.randn(784,10).astype(np.float32))
B=tf.Variable(0.1*np.random.randn(10).astype(np.float32))

pred = tf.nn.softmax(tf.add(tf.matmul(X,W),B))
#cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),axis=1))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as sesh:
	sesh.run(init)
	
	y_train=sesh.run(tf.one_hot(y_train,10))
	y_test=sesh.run(tf.one_hot(y_test,10))
	
	
	for epoch in range(epochs):
		for i in range(batches):
			offset = i * epoch
			x = x_train[offset: offset + batch_size]
			y = y_train[offset: offset + batch_size]
			sesh.run(optimizer, feed_dict={X: x,Y: y})
			c = sesh.run(cost, feed_dict={X: x,Y: y})
		if not epoch %1:
			print(f'epoch:{epoch} cost={c:.4f}')
			
			
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y,1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	acc = accuracy.eval({X: x_test, Y:y_test})
	print(acc)
	
	fig, axes = plt.subplots(1,10, figsize=(8,4))
	for img, ax in zip(x_test[:10], axes):
		guess=np.argmax(sesh.run(pred, feed_dict={X: [img]}))
		ax.set_title(guess)
		ax.imshow(img.reshape((28,28)))
		ax.axis('off')
plt.show()	
