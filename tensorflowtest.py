import tensorflow as tf

x = tf.placeholder(tf.float32, shape = [None, 1])
W1 = tf.constant(2, dtype = tf.float32, shape = [1,1])
y = tf.matmul(x, W1)

sess1 = tf.Session()
sess1.run(tf.global_variables_initializer())

print(sess1.run(W1))

print(sess1.run(y, feed_dict={x : [[1],[2],[3],[4]]}))

sess1.close()

with tf.Session() as sess:
	print(sess.run(W1))

	print(sess.run(y, feed_dict={x : [[1],[2],[3],[4]]}))
