import tensorflow as tf

class NeuralLM(object):
	"""docstring for NeuralLM"""
	def __init__(self, arg):
		super(NeuralLM, self).__init__()
		self.arg = arg

		maxContentLength = arg['maxContentLength']
		vocabSize = arg['vocabSize']
		embDim = arg['embDim']
		rnnLayers = arg['rnnLayers']
		rnnSize = arg['rnnSize']
		nbClasses = arg['nbClasses']
		topK = arg['topK']
		maxGradNorm = arg['maxGradNorm']

		#Placeholders for input
		self.x = tf.placeholder(tf.int32, [None, None], name="inputWordIndices")
		self.xLen = tf.placeholder(tf.int32, [None], name='inputContextLengths')
		self.y = tf.placeholder(tf.int32, [None, None], name='targetWordIndices')

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
			contextEmbedded, _ = tf.nn.dynamic_rnn(cell=rnnMultiCells, inputs=wordEmbedded, sequence_length=self.xLen, dtype=tf.float32)

			self.rnnVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "rnnWordLayer")

		#Feed-Forward Layer
		with tf.variable_scope("ffTargetLayer") as ffVS:
			W = tf.get_variable("ffTargetWeight", shape=[rnnSize, nbClasses], dtype=tf.float32)
			b = tf.get_variable("ffTargetBias", shape=[nbClasses], dtype=tf.float32)

			originalShape = tf.shape(contextEmbedded)
			contextEmbedded2D = tf.reshape(contextEmbedded, [-1, rnnSize])  # [BatchSize, InputLength, RNNSize] -> [BatchSize * InputLength, RNNSize]

			logits = tf.nn.xw_plus_b(contextEmbedded2D, W, b, name="feedforwardTargerLayer")

			logits = tf.reshape(logits, originalShape) # [BatchSize * InputLength, RNNSize] -> [BatchSize, InputLength, RNNSize]

			self.topKPred = tf.nn.top_k(logits, k=topK, name="topKPredictions")
			self.predictions = tf.argmax(logits, 2, name="predictions")  # return type = tf.int64
			self.ffVariables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="ffTargetLayer")

		#Loss
		with tf.variable_scope("loss"):
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y, name='losses')
			lossMask = tf.sequence_mask(self.xLen, maxlen=maxContentLength)
			losses = tf.contrib.seq2seq.sequence_loss(logits, self.y, weights=lossMask, average_across_timesteps=False, average_across_batch=True)
			self.cost = tf.reduce_sum(losses, name="cost")

		#Accuracy
		with tf.variable_scope("accuracy"):
			nbCorrects = tf.equal(tf.cast(self.predictions, tf.int32), self.y)
			self.accuracy = tf.reduce_mean(tf.cast(nbCorrects, tf.float32))
			#TODO: Get TopK Accuracy

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

		#Summaries -> Tensorbroad
		costSummary = tf.summary.scalar("costSummary", self.cost)
		accSummary = tf.summary.scalar("accSummary", self.accuracy)
		self.summaryOp = tf.summary.merge([costSummary, accSummary])


	def assignLearningRate(self, sess, lrValue):
		# sess.run(sefl.updateLr, feed_dict={self.newLr: lrValue})
		sess.run(self.lr.assign(lrValue))


	def assignPretrainWordEmbedding(self, sess, wordEmbedding):
		sess.run(self.embedding.assign(wordEmbedding))