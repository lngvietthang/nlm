import tensorflow as tf

class NeuralLM(object):
	"""docstring for NeuralLM"""
	def __init__(self, arg):
		super(NeuralLM, self).__init__()
		self.arg = arg

		#Placeholders for input
		self.x = tf.placeholder(tf.int32, [None, maxContentLength], name="inputWordIndices")
		self.xLen = tf.placeholder(tf.int32, [None], name='inputContextLengths')
		self.y = tf.placeholder(tf.int32, [None], name='targetWordIndices')
		# self.newLr = tf.placeholder(tf.float32, [], name="newLearningRate")

		#Embedding Layer
		with tf.variable_scope("wordEmbeddingLayer"):
			with tf.device("/cpu:0"):
				self.embedding = tf.get_variable("wordEmbeddingMatrix", [vocabSize, embDim], dtype=tf.float32)
				wordEmbedded = tf.nn.embedding_lookup(self.embedding, self.x, name="wordEmbeddedVectors")

		#TODO: Dropout
		#...

		#RNN Graph
		with tf.variable_scope("rnnWordLayer") as rnnVS:
			rnnCells = []
			for i in range(rnnLayers):
				rnnCells.append(tf.contrib.rnn.LSTMBlockCell(rnnSize, name="rnnLayer-{}".format(i+1)))
			rnnMultiCells = tf.contrib.rnn.MultiRNNCell(rnnCells)
			contextEmbedded = tf.nn.dynamic_rnn(cell=rnnMultiCells, inputs=wordEmbedded, sequence_length=self.xLen, dtype=tf.float32)
			self.rnnVariables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=rnnVS)

			batchRange = tf.range(tf.shape(contextEmbedded)[0])  # <- batchSize: [0, 1, ..., batchSize - 1]
			indices = tf.stack([batchRange, self.xLen - 1], axis=1) # indices: [[0, L[0] - 1], [1, L[1] - 1], ..., [batchSize - 1, L[batchSize-1] -1]]
			contextEncoded = tf.gather_nd(contextEmbedded, indices)

		#Feed-Forward Layer
		with tf.get_variable_scope("ffTargetLayer") as ffVS:
			W = tf.get_variable("ffTargetWeight", shape=[rnnSize, nbClasses], dtype=tf.float32)
			b = tf.get_variable("ffTargetBias", shape=[nbClasses], dtype=tf.float32)

			logits = tf.nn.xw_plus_b(contextEncoded, self.W, self.b, name="feedforwardTargerLayer")

			self.topKPred = tf.nn.top_k(logits, k=3, name="topKPredictions")  #TODO: set K outside
			self.predictions = tf.argmax(logits, 1, name="predictions")  # return type = tf.int64
			self.ffVariables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=ffVS)

		#Loss
		with tf.get_variable_scope("loss"):
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y, name='losses')
			self.cost = tf.reduce_mean(losses, name="cost")  # Use mean to make the gradient magnitude independent with batch size

		#Accuracy
		with tf.get_variable_scope("accuracy"):
			nbCorrects = tf.equal(tf.cast(self.predictions, tf.int32), self.y)
			self.accuracy = tf.reduce_mean(tf.cast(nbCorrects, tf.float32))

		#
		self.globalStep = tf.Variable(0, name='globalStep', trainable=False)
		self.lr = tf.Variable(0.0, trainable=False)

		#Get weights of computation graph
		tvars = tf.trainable_variables()

		#Optimizer
		self.optimizer = tf.train.GradientDescentOptimizer(self.lr)

		#Calculating gradients and clipping by global norm if gradient over maxGradNorm
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), maxGradNorm)

		#Train
		self.trainOp = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.globalStep)

		# #Learning rate update
		# self.updateLr = tf.assign(self.lr, self.newLr)

	def assignLearningRate(self, sess, lrValue):
		# sess.run(sefl.updateLr, feed_dict={self.newLr: lrValue})
		sess.run(self.lr.assign(lrValue))


	def assignPretrainWordEmbedding(self, sess, wordEmbedding):
		sess.run(self.embedding.assign(wordEmbedding))