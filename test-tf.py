import tensorflow as tf
import numpy as np

rvA = np.random.randn(1, 5, 4)  # Padding
rvB = np.random.randn(1, 5, 4)

print rvA.shape
print rvA
print rvB.shape
print rvB


rvC = np.concatenate((rvA, rvB), axis=0)
print rvC.shape
print rvC

rvL = np.array([3, 5])
print rvL.shape
print rvL

##Build TF Graph
# Static RNN
# X = tf.placeholder(tf.float32, shape=[None, 5, 4], name='input')
# L = tf.placeholder(tf.int32, shape=[None], name='length')
# rnnInput = tf.unstack(X, axis=1)
# rnnCell = tf.nn.rnn_cell.BasicRNNCell(num_units=3)

# rnnOut, rnnState = tf.nn.static_rnn(rnnCell, rnnInput, sequence_length=L, dtype=tf.float32)

# Dynamic RNN
X = tf.placeholder(tf.float32, shape=[None, 5, 4], name='input')
rnnInput = X
L = tf.placeholder(tf.int32, shape=[None], name='length')
rnnCell = tf.nn.rnn_cell.BasicRNNCell(num_units=3)
rnnOut, rnnState = tf.nn.dynamic_rnn(rnnCell, rnnInput, sequence_length=L, dtype=tf.float32)

# Get last output use tf.gather_nd
batchRange = tf.range(tf.shape(rnnOut)[0])  # <- batchSize: [0, 1, ..., batchSize - 1]
indices = tf.stack([batchRange, L - 1], axis=1) # indices: [[0, L[0] - 1], [1, L[1] - 1], ..., [batchSize - 1, L[batchSize-1] -1]]
rnnLastOut = tf.gather_nd(rnnOut, indices)

#
rnnOut2D = tf.reshape(rnnOut, [-1, 3])

w = tf.get_variable('w', [3, 4], dtype=tf.float32)
b = tf.get_variable('b', [4], dtype=tf.float32)
ffOut = tf.nn.xw_plus_b(rnnOut2D, w, b)

ffOut3D = tf.reshape(ffOut, [2, 5, 4])

timelineRange = tf.range(L)
indices = tf.stack([batchRange, 0 : L - 1], axis=1)
ffOut3D = tf.gather_nd(ffOut3D, indices)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# a = sess.run(rnnInput, feed_dict={X: rvC})
	a, b, s, c = sess.run([rnnInput, rnnOut, rnnOut2D, ffOut3D], feed_dict={X: rvC, L: rvL})

	# print a.shape
	for item in a:
		print "--------------------"
		print item.shape
		print item
	# print b.shape
	for item in b:
		print "++++++++++++++++++++"
		print item.shape
		print item

	for item in s:
		print "===================="
		print item.shape
		print item

	for item in c:
		print "********************"
		print item.shape
		print item