#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import copy

#normalizes a move to only contain one 1 and eight 0
def norm_move(move):
	out = np.zeros([9])
	out[np.argmax(move)] = 1
	return out

#Returns alowed moves in a numpy array
def alowed_moves(board):
	return np.array(list(map(lambda x : ((x**2)-1)**2, board)))

#Checks if a move is alowed
def check_move(board, move):
	return (alowed_moves(board) == ((move + 2) / 3)).any()

#Combindes the board with a move
def make_move(board, move):
	out = copy.copy(board)
	out[np.argmax(move)] = 1.
	return out

#Checks if there is a win
def check_win(board):
	win = False
	
	f = np.reshape(board, [3, 3])	
	for x in range(3):
		if (np.unique(f[x]) == [1]).all() or (np.unique(f.T[x]) == [1]).all():
			win = True
	
	r = np.argmax(board)
	if r % 2 == 0 and board[4] == 1 and win == False:
		if board[0] == 1 == board[8]:
			win = True
		if board[2] == 1 == board[6]:
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
board = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
test = sess.run(y, feed_dict={x: [board]})
ntest = norm_move(test)
nboard = make_move(board, ntest)
print("Board:", board)
print("Move:", test)
print("Norm move:", ntest)
print("Alowed:", alowed_moves(board))
print("Check move:", check_move(board, ntest))
print("New board:", nboard)
print(np.reshape(nboard, [3, 3]))
print("Check win:", check_win(nboard))

#Close session
sess.close()
