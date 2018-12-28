#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

#normalizes a move to only contain one 1 and eight 0
def norm_move(move):
	out = np.zeros([9])
	out[np.argmax(move)] = 1
	return out

#Returns alowed moves in a numpy array
def alowed_moves(field):
	return np.array(list(map(lambda x : ((x**2)-1)**2, field)))

#Checks if a move is alowed
def check_move(field, move):
	return (alowed_moves(field) == move).any()

#Checks if there is a win
def check_win(field):
	win = False
	
	f = np.reshape(field, [3, 3])	
	for x in range(3):
		if (np.unique(f[x]) == [1]).all() or (np.unique(f.T[x]) == [1]).all():
			win = True
	
	r = np.argmax(field)
	if r % 2 == 0 and field[4] == 1 and win == False:
		if field[0] == 1 == field[8]:
			win = True
		if field[2] == 1 == field[6]:
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
board = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
test = sess.run(y, feed_dict={x: [board]})
ntest = norm_move(test)
print("Move:", test)
print("Norm move:", ntest)
print("Alowed:", alowed_moves(board))
print("Check move:", check_move(board, ntest))
print("Check win:", check_win(ntest))

#Close session
sess.close()
