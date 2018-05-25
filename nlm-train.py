import os
import datetime
import time

import datetime
import tensorflow as tf

import utils

from nlm import NeuralLM


def run_epoch(sess, model, data, batchSize, summaryWriter, training=True, verbose=False):
	
	totalIns = x.shape[0]
	nbBatch = totalIns / batchSize + (totalIns % batchSize > 0)

	totalCosts = 0.0
	accuracy = 0.0

	fetchs = {
		'globalStep': model.globalStep,
		'cost': model.cost,
		'accuracy': model.accuracy,
		'summary': model.summaryOp
	}

	if trainning:
		fetchs['training'] = model.trainOp

	bStart = 0
	step = 0
	for i in range(nbBatch):
		#
		bEnd = min(bStart+batchSize, totalIns)
		bX = x[bStart : bEnd]
		bXLen = xLen[bStart : bEnd]
		nY = y[bStart: bEnd]

		feedDict = {
			model.x: bX,
			model.xLen: bXLen,
			model.y: nY
		}
		vals = sess.run(fetchs, feedDict)

		step, bCost, bAccuracy, bSummaries = vals['globalStep'], vals['cost'], vals['accuracy'], vals['summary']

		summaryWriter.add_summary(bSummaries, step)

		if verbose and ((i + 1) % 50 == 0):
			timeStr = datetime.datetime.now().isoformat()
			print "--> {}: step {}, cost: {:g}, acc: {:g}".format(timeStr, step, bCost, bAccuracy)

		bStart = bEnd


	return step


def evaluation_run(sess, step, model, data, summaryWriter, verbose=False):
	x, xLen, y = data

	fetchs = {
		'cost': model.cost,
		'accuracy': model.accuracy,
		'summary': model.summaryOp
	}

	feedDict = {
		model.x: x,
		model.xLen: xLen,
		model.y: y
	}
	vals = sess.run(fetchs, feedDict)

	cost, accuracy, summaries = vals['cost'], vals['accuracy'], vals['summary']

	summaryWriter.add_summary(summaries, step)

	if verbose:
		timeStr = datetime.datetime.now().isoformat()
		print "--> {}: step {}, cost: {:g}, acc: {:g}".format(timeStr, step, cost, accuracy)


def main():
	path2trainData = "xxxxx"
	path2devData = "xxxxx"
	path2pretrainWordEmbedding = "xxxxx"
	path2config = "xxxxx"

	# Training Config
	nbEpoch = 100
	batchSize = 512
	lr = 1
	lrDecay = 0.5
	nbEpochBeforeLRDecay = 4

	# TensorBroad: Save summaries
	timestamp = str(int(time.time()))
	path2out = os.path.join("xxxxx", timestamp)
	if not os.path.exists(path2out):
		os.makedirs(path2out)
	path2logDir = os.path.join(path2out, "logs")
	path2ckp = os.path.join(path2out, "checkpoints")
	ckpPrefix = os.path.join(path2ckp, "NeuralLM")
	os.makedirs(path2ckp)

	# Model checkpoint config
	nbRecentCkpKeep = 2

	# modelConfig = {
	# 	'maxContentLength': 6,
	# 	'vocabSize': 3000,
	# 	'embDim': 100,
	# 	'rnnLayers': 2,
	# 	'rnnSize': 512,
	# 	'nbClasses': 3000,
	# 	'topK': 3
	# }
	modelConfig = utils.loadModelConfig(path2config)

	trainData = utils.loadData(path2trainData)
	devData = utils.loadData(path2devData)

	pretrainWordEmbedding = utils.loadPretrainWordEmbedding(path2pretrainWordEmbedding)

	with tf.Graph().as_default():
		# Ref: https://stackoverflow.com/questions/44873273/what-do-the-options-in-configproto-like-allow-soft-placement-and-log-device-plac
		sessionConfig = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)
		sess = tf.Session(config=sessionConfig)
		with sess.as_default():
			neuralLM = NeuralLM(modelConfig)

			trainSummaryWriter = tf.summary.FileWriter(os.path.join(path2logDir, 'train'), sess.graph)
			devSummaryWriter = tf.summary.FileWriter(os.path.join(path2logDir, 'development'), sess.graph)

			saver = tf.train.Saver(tf.global_variables(), max_to_keep=nbRecentCkpKeep)

			# Initialize all variables
			sess.run(tf.global_variables_initializer())

			# Assign pretrain word embedding to embedding layer of model
			neuralLM.assignPretrainWordEmbedding(sess, pretrainWordEmbedding)

			for e in range(nbEpoch):
				newLRDecay = lrDecay ** max(e + 1 - nbEpochBeforeLRDecay, 0.0)
				neuralLM.assignLearningRate(sess, lr * newLRDecay)

				# Training progress
				print "-> Epoch {}: Training Progress".format(e)
				currStep = run_epoch(sess, neuralLM, trainData, batchSize, trainSummaryWriter, verbose=True)

				# Development progress
				print "-> Epoch {}: Development Progress".format(e)
				evaluation_run(sess, currStep, neuralLM, devData, devSummmaryWriter, verbose=True)

				# Save checkpoint
				path = saver.save(sess, ckpPrefix, global_step=currStep)
				print "-> Epoch {}: Saved Checkpoint - {}".format(e, path)

				# TODO: Early stopping







if __name__ == '__main__':
	main()