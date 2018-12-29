#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from copy import copy
import random

#normalizes a move to only contain one 1 and eight 0, return 2 moves with the highest probability
def norm_move(move):
	temp = copy(move)
	out1 = np.zeros([9])
	out1[np.argmax(temp)] = 1
	temp[0][np.argmax(temp)] = 0
	out2 = np.zeros([9])
	out2[np.argmax(temp)] = 1
	return out1, out2

#Returns alowed moves in a numpy array
def alowed_moves(board):
	return np.array(list(map(lambda x : ((x**2)-1)**2, board)))

#Checks if a move is alowed
def check_move(board, move):
	return (alowed_moves(board) == ((move + 2) / 3)).any()

#Combindes the board with a move
def make_move(board, move):
	out = copy(board)
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

def non_losing_moves(board, lmove):
	out = alowed_moves(board)
	out[np.argmax(lmove)] = 0.
	return out

#Define Placeholders
x = tf.placeholder(tf.float32, shape=[None, 9], name = "x")
y_ = tf.placeholder(tf.float32, shape=[None, 9], name = "y_")

#Define Weights
W1 = tf.Variable(tf.random_normal(shape=[9, 32]))
W2 = tf.Variable(tf.random_normal(shape=[32, 9]))

b1 = tf.Variable(tf.random_normal(shape=[9]))

#Define Graph
y = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(x, W1)), W2) + b1, name = "y")

###########
loss = tf.losses.softmax_cross_entropy(y_, y)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

#Initialize a session and variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Training
for game in range(100000):
	board = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
	moves = [[], []]#move[0] is the inputed board layout, move[1] is the outputed move
	win = False
	turn = 0
	inputs = []
	lables = []
	for t in range(9):
		turn = t
		move1, move2 = norm_move(sess.run(y, feed_dict={x: [board]}))
		for move in [move1, move2]:
			if check_move(board, move):
				moves[0].append(board)
				moves[1].append(move)
				board = make_move(board, move)
				#print(np.reshape(move, [3,3]), "m")
				break
			else:
				if (move == move2).all():
					alowed = alowed_moves(board)
					i_alowed = [index for index, value in enumerate(alowed) if value == 1]
					#rmove = alowed[random.choice(i_alowed)]###WRONG
					rmove = np.zeros([9])
					rmove[random.choice(i_alowed)] = 1
					moves[0].append(board)
					moves[1].append(rmove)
					board = make_move(board, rmove)
					#print(np.reshape(rmove, [3,3]), "r")
		if check_win(board):
			win = True
			#print("-"*10, "Turn:", turn, "Game:", game)
			#print(np.reshape(board, [3, 3]) *(-1/(((turn % 2)*2)-1)), "\n")
			break
		#print("-"*10, "Turn:", turn, "Game:", game)
		#print(np.reshape(board, [3, 3]) *(-1/(((turn % 2)*2)-1)), "\n")
		board *= -1
			
	if win:
		inputs.extend(moves[0][turn % 2 :: 2])#Wining moves
		lables.extend(moves[1][turn % 2 :: 2])#
		for index in range((turn + 1) % 2, turn + 1, 2):
			inputs.append(moves[0][index])
			lables.append(non_losing_moves(moves[0][index], moves[1][index]))
	else:
		for index in range(0, turn + 1):
			inputs.append(moves[0][index])
			lables.append(non_losing_moves(moves[0][index], moves[1][index]))

	#print(sess.run(loss,  feed_dict={y_: np.array(lables), x: np.array(inputs)}))
	sess.run(train, feed_dict={y_: np.array(lables), x: np.array(inputs)})
	#print(sess.run(loss,  feed_dict={y_: np.array(lables), x: np.array(inputs)}))
	#print("-"*10)

#Some testing
board = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.])
test = sess.run(y, feed_dict={x: [board]})
ntest = norm_move(test)[0]
#nboard = make_move(board, ntest)
print("Board:", board)
print("Move:", test)
print("Norm move:", ntest)
#print("Alowed:", alowed_moves(board))
#print("Check move:", check_move(board, ntest))
#print("New board:", nboard)
#print(np.reshape(nboard, [3, 3]))
#print("Check win:", check_win(nboard))

#Close session
sess.close()
