import numpy as np
import BotAPI
import argparse
from copy import copy

#Returns alowed moves in a numpy array
def alowed_moves(board):
	return np.array(list(map(lambda x : ((x**2)-1)**2, np.reshape(board, [9]))))


#Combindes the board with a move
def make_move(board, move):
	#print("-"*10)
	#print(board, "Board")
	#print(move, "move")
	out = np.reshape(board, [9])
	out[np.argmax(move)] = 1.
	#print("-"*10)
	#print(board, "Board")
	#print(move, "move")
	#print("-"*10)
	return np.reshape(out, [3, 3])

#Checks if there is a win
def check_win(board):
	win = False

	for x in range(3):
		if (np.unique(board[x]) == [1]).all() or (np.unique(board.T[x]) == [1]).all():
			win = True

	f = np.reshape(board, [9])
	r = np.argmax(f)
	if r % 2 == 0 and f[4] == 1 and win == False:
		if f[0] == 1 == f[8]:
			win = True
		if f[2] == 1 == f[6]:
			win = True

	return win

#Returns all possible moves exept the one specified
def non_losing_moves(board, lmove):
	out = alowed_moves(np.reshape(board, [9]))
	out[np.argmax(np.reshape(lmove, [9]))] = 0.
	return np.reshape(out, [3, 3])


parser = argparse.ArgumentParser()

parser.add_argument("--bots", "-b", help='Specify witch bot/bots used in training. To specify two bots for training their names should be seperated by a ", ". E.g: -b "bot1, bot2". If ony one bot is specified then it will play agains it self. If no bot is specified then Charlie will be used during the training session.')

parser.add_argument("--games", "-g", help='Specify the number of games to be played during the training session. If games is not specified the number of games will be 20.')

# read arguments from the command line
args = parser.parse_args()

bots = ["Charlie"]
if args.bots:
	bots = (args.bots).split(", ")

for b in bots:
	print(b, "selected")

games = 20
if args.games:
	games = args.games
print(games, "games will be played")



c = BotAPI.Bot()
if c.LoadBot(bots[0]) == False:
	c.NewBot(bots[0])

if len(bots) < 1:
	d = BotAPI.Bot()
	if d.LoadBot(bots[1]) == False:
		d.NewBot(bots[1])


for game in range(int(games)):
	board = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
	moves = [[], []]#move[0] is the inputed board layout, move[1] is the outputed move
	win = False
	turn = 0
	inputs = []
	lables = []
	for t in range(9):
		turn = t
		if (game + t) % 2 == 0:
			move = c.MakeMove(board)
			moves[0].append(copy(board))
			moves[1].append(copy(move))
			#print("-"*20)
			#print(board, "r")
			make_move(board, move)#####HOW!!!
			#print(board, "r")
			#print("-"*20)
		else:
			if len(bots) < 1:
				move = d.MakeMove(board)
			else:
				move = c.MakeMove(board)
			moves[0].append(copy(board))
			moves[1].append(copy(move))
			#print("-"*20)
			#print(board, "r")
			make_move(board, move)#####HOW!!!
			#print(board, "r")
			#print("-"*20)

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

	#for te in range(len(inputs)):
	#	print(inputs[te])
	#	print(lables[te])
	#print(moves)
	c.Train(inputs, lables, save = False, log = False)

	if len(bots) < 1:
		d.Train(inputs, lables, save = False, log = False)
	#print(game)
	#print(sess.run(loss,  feed_dict={y_: np.array(lables), x: np.array(inputs)}))
	#sess.run(train, feed_dict={y_: np.array(lables), x: np.array(inputs)})
	#print(sess.run(loss,  feed_dict={y_: np.array(lables), x: np.array(inputs)}))
	#print("-"*10)

print(c.name, "trained and saved at", c.Save())
if len(bots) < 1:
	print(d.name, "trained and saved at", d.Save())
