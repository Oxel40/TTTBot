#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

def norm_move(move):
	out = np.zeros([9])
	out[np.argmax(move)] = 1
	return out

def alowed_moves(field):
	return np.array(list(map(lambda x : ((x**2)-1)**2, z)))

def check_move(field, move):
	return (alowed_moves(field) == move).any()

def check_win(field, move):
#Move needs to be included in field before calling this funktion! The move parameter is only used to decide where to look for a win.
	win = False
	
	f = np.reshape(field, [3, 3])
	r = np.argmax(move)	
	if (np.unique(f[r%3]) == [1]).all() or (np.unique(f.T[r//3]) == [1]).all():
		win = True
	
	if r % 2 == 0 and field[4] == 1 and win == False:
		if field[0] == 1 == field[6]:
			win = True
		if field[2] == 1 == field[8]:
			win = True

	return win

#Define Placeholders
x = tf.placeholder(tf.float32, shape=[None, 9])
y_ = tf.placeholder(tf.float32, shape=[None, 9])

#Define Weights
W1 = tf.Variable(tf.random_normal(shape=[9, 32]))
W2 = tf.Variable(tf.random_normal(shape=[32, 9]))

b1 = tf.Variable(tf.random_normal(shape=[9]))

#Define Graph
y = tf.nn.tanh(tf.matmul(tf.nn.tanh(tf.matmul(x, W1)), W2) + b1)

#Initialize a session and variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Some testing
print(sess.run(y, feed_dict={x: [[0,0,0,0,0,0,0,0,0]]}))
print(np.argmax(sess.run(y, feed_dict={x: [[0,0,0,0,0,0,0,0,0]]})))
print(sess.run(y, feed_dict={x: [[0,0,0,0,-1,-1,1,0,0]]}))
print(np.argmax(sess.run(y, feed_dict={x: [[0,0,0,0,-1,-1,1,0,0]]})))

#Close session
sess.close()
