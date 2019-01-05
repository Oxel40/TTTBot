import tensorflow as tf

class test1:
	def __init__(self):
		self.sess = tf.Session()

		self.x = tf.placeholder(tf.float32, shape = [None, 1])
		self.W1 = tf.constant(2, dtype = tf.float32, shape = [1,1])
		self.y = tf.matmul(self.x, self.W1)
		
		self.sess.run(tf.global_variables_initializer())
	
	def run(self):
		print(self.sess.run(self.y, feed_dict = {self.x: [[1], [2], [3], [4]]}))
	
	def end(self):
		self.sess.close()

class test2:
	with tf.Session() as sess:
		def __init__(self):
			self.x = tf.placeholder(tf.float32, shape = [None, 1])
			self.W1 = tf.constant(3, dtype = tf.float32, shape = [1,1])
			self.y = tf.matmul(self.x, self.W1)
			
			self.sess.run(tf.global_variables_initializer())
		
		def run(self):
			print(self.sess.run(self.y, feed_dict = {self.x: [[1], [2], [3], [4]]}))


e = test1()
e.run()
e.end()

q = test2()
q.run()

