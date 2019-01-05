import tensorflow as tf
import numpy as np
import os
from copy import copy

class Bot:
	def __init__(self):
		self.sess = tf.Session()
		self.name = None
		self.Decision = []


	def NewBot(self, new_name):
		#NOTE: This funktion overwrites any existing models with same name!
		self.name = new_name

		#Define Placeholders
		self.x = tf.placeholder(tf.float32, shape=[None, 9], name = self.name+"x")
		self.y_ = tf.placeholder(tf.float32, shape=[None, 9], name = self.name+"y_")

		#Define Weights
		self.W1 = tf.Variable(tf.random_normal(shape=[9, 32]), name = self.name+"W1")
		self.W2 = tf.Variable(tf.random_normal(shape=[32, 9]), name = self.name+"W2")

		self.b1 = tf.Variable(tf.random_normal(shape=[9]), name = self.name+"b1")

		#Define Graph
		self.y = tf.nn.sigmoid(tf.matmul(tf.nn.tanh(tf.matmul(self.x, self.W1)), self.W2) + self.b1, name = self.name+"y")

		#Initialize Graph
		self.sess.run(tf.global_variables_initializer())

		save_path = self.Save(meta = True)
		print("{0} created and saved in path: {1}".format(self.name, save_path))


	def LoadBot(self, name):
		#NOTE: Returns a True or False depending on whether it could load the model.
		try:
			self.name = name
			saver = tf.train.import_meta_graph("model/{0}/{0}.meta".format(self.name))
			saver.restore(self.sess, "model/{0}/{0}".format(self.name))
			print("{0} loaded from path: model/{0}/{0}".format(self.name))
			del saver
			return True
		except:
			print("Could not locate or restore {0}. Make sure the directory model/{0}/ exists and that it contains all necessary files".format(name))
			self.name = None
			return False


	def MakeMove(self, board):
		#NOTE: If no alowed move is found then this funktion returns a 3x3 array of zeros.
		found = False
		board = np.reshape(board, [9])
		alowed = np.array(list(map(lambda x : ((x**2)-1)**2, board)))
		pred = self.sess.run(self.name+"y:0", feed_dict={self.name+"x:0": np.reshape(board, [1, 9])})
		pred = np.reshape(pred, [9])

		for n in range(9):
			move = np.zeros([9])
			move[np.argmax(pred)] = 1
			if (alowed == ((move + 2) / 3)).any():
				found = True
				break
			pred[np.argmax(pred)] = 0

		move = np.reshape(move, [3, 3])
		return move if found else np.zeros([3, 3])


	def Train(self, data, labels, save = True, log = True, rate = 0.1):
		ndata = []
		nlabels = []
		for n in data:
			ndata.append(np.reshape(n, [9]))

		for n in labels:
			nlabels.append(np.reshape(n, [9]))

		graph = tf.get_default_graph()
		y_ = graph.get_tensor_by_name(self.name+"y_:0")
		y = graph.get_tensor_by_name(self.name+"y:0")

		loss = tf.losses.softmax_cross_entropy(y_, y)
		optimizer = tf.train.GradientDescentOptimizer(rate)
		train = optimizer.minimize(loss)

		self.sess.run(train, feed_dict={self.name+"y_:0": np.array(nlabels), self.name+"x:0": np.array(ndata)})

		if save and log:
			save_path = self.Save()
			print("{0} trained and saved in path: {1}".format(self.name, save_path))
		elif save:
			save_path = self.Save()
		elif log:
			print("{0} trained".format(self.name))


	def Save(self, meta = False):
		try:
			os.mkdir("model")
			saver = tf.train.Saver(save_relative_paths = True)
			save_path = saver.save(self.sess, "model/{0}/{0}".format(self.name), write_meta_graph = meta)
			del saver
		except FileExistsError:
			saver = tf.train.Saver(save_relative_paths = True)
			save_path = saver.save(self.sess, "model/{0}/{0}".format(self.name), write_meta_graph = meta)
			del saver

		return save_path


	def Click_Coords(self, GameMatrixState):
		move = self.MakeMove(GameMatrixState)
		#Get the coordinates of the element with the
		#highest probability value
		clickCords = self.MaxCords(move)
		#collect some data in the process
		decision = np.zeros((3,3))
		decision[clickCords[0],clickCords[1]] = 1
		self.Decision.append((copy(GameMatrixState), decision))
		return clickCords


	def MaxCords(self, GuessMatrix):
				max = 0
				maxcords = (0,0)
				for i in range(3):
					for j in range(3):
						if GuessMatrix[i,j] > max:
							maxcords = (i,j)
							max = GuessMatrix[i,j]
				return maxcords


	def ListBots(self):
		return list(os.walk("model"))[0][1]
	
	
	def Close(self):
		#NOTE: This should always be called after all the computations are done! 
		self.sess.close()


if __name__ == "__main__":
	e = Bot()
	q = Bot()
	q.NewBot("Billy")
	e.LoadBot("Tom")
	#print(e.MakeMove(np.array([[1, -1, 1], [-1, 1, -1], [-1, 1, -1]])))
	boards = [np.array([[0, 0, 0], [0, 0, 0], [ 0, 0, 0]]), np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])]
	moves = [np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])]
	e.Train(boards, moves)
	q.Train(board, moves)
	#print(e.ListBots())
	e.Close()
	q.Close()
