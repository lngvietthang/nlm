import os
import time
import math
import datetime
import random

import sys
import codecs
sys.stdout=codecs.getwriter('utf-8')(sys.stdout)  # Print unicode to console logging (Ref: https://stackoverflow.com/questions/5530708/can-i-redirect-unicode-output-from-the-console-directly-into-a-file)

import datetime
import tensorflow as tf

import utils

from nlm import NeuralLM


def stepDecayLearningRate(epoch, initLR, dropRate, nbEpochsDrop, minimumLR):
	# Ref: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
	# LearningRate = InitialLearningRate * DropRate^floor(Epoch / EpochDrop)
	newLR = initLR * math.pow(dropRate, math.floor((1 + epoch) / nbEpochsDrop))
	if newLR > minimumLR: 
		return newLR
	else:
		return minimumLR



def run_epoch(sess, model, data, batchSize, summaryWriter, trainning=True, verbose=False, debug=False):
	x, xLen, y = data

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

	if debug:
		fetchs['topK'] = model.topKPred

	bStart = 0
	step = 0
	print 'Number of Batches:', nbBatch
	for i in range(nbBatch):
		#
		bEnd = min(bStart+batchSize, totalIns)
		bX = x[bStart : bEnd]
		bXLen = xLen[bStart : bEnd]
		bY = y[bStart: bEnd]

		feedDict = {
			model.x: bX,
			model.xLen: bXLen,
			model.y: bY
		}
		vals = sess.run(fetchs, feedDict)

		step, bCost, bAccuracy, bSummaries = vals['globalStep'], vals['cost'], vals['accuracy'], vals['summary']

		summaryWriter.add_summary(bSummaries, step)

		if verbose and ((i + 1) % 50 == 0):
			timeStr = datetime.datetime.now().isoformat()
			print "--> {}: step {}, cost: {:g}, acc: {:g}".format(timeStr, step, bCost, bAccuracy)

		if debug and ((i + 1) % 50 == 0):
			timeStr = datetime.datetime.now().isoformat()
			print "--> {}: step {} DEBUG Info".format(timeStr, step)
			topK = vals['topK']
			currBatchSize = bX.shape[0]
			randomIndices = range(currBatchSize)
			random.shuffle(randomIndices)
			randomIndices = randomIndices[:5]
			invertVocab = utils.loadPKLFile('data/nlm-input/invert-vocab.pkl')

			for index in randomIndices:
				context = utils.mapIndex2Word(bX[index], invertVocab)
				target = utils.mapIndex2Word([bY[index]], invertVocab)
				pred = utils.mapIndex2Word(topK[1][index], invertVocab)
				print u"\t\t{} -> {} | {}".format(u" ".join(context), target[0], u",".join(pred))
				

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

	return cost, accuracy


def main():
	path2data = "data/nlm-input/"
	path2config = "NeuralLM.config"

	# Training Config
	nbEpoch = 30
	batchSize = 512
	initLR = 1.0
	minimumLR = 0.0001
	dropRateLR = 0.5
	nbEpochsDropLR = 3

	# TensorBroad: Save summaries
	timestamp = str(int(time.time()))
	path2out = os.path.join("out/", timestamp)
	if not os.path.exists(path2out):
		os.makedirs(path2out)
	path2logDir = os.path.join(path2out, "logs")
	path2ckp = os.path.join(path2out, "checkpoints")
	ckpPrefix = os.path.join(path2ckp, "NeuralLM")
	os.makedirs(path2ckp)

	# Model checkpoint config
	nbRecentCkpKeep = 2

	modelConfig = utils.loadModelConfig(path2config)

	trainData, wordEmbedding = utils.loadData(path2data, "train", loadWordEmbedding=True)
	devData = utils.loadData(path2data, "dev")

	with tf.Graph().as_default():
		# Ref: https://stackoverflow.com/questions/44873273/what-do-the-options-in-configproto-like-allow-soft-placement-and-log-device-plac
		sessionConfig = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
		sess = tf.Session(config=sessionConfig)
		with sess.as_default():
			neuralLM = NeuralLM(modelConfig)

			trainSummaryWriter = tf.summary.FileWriter(os.path.join(path2logDir, 'train'), sess.graph)
			devSummaryWriter = tf.summary.FileWriter(os.path.join(path2logDir, 'development'), sess.graph)

			saver = tf.train.Saver(tf.global_variables(), max_to_keep=nbRecentCkpKeep)

			# Initialize all variables
			sess.run(tf.global_variables_initializer())

			# Assign pretrain word embedding to embedding layer of model
			neuralLM.assignPretrainWordEmbedding(sess, wordEmbedding)

			for e in range(nbEpoch):
				newLR = stepDecayLearningRate(e, initLR, dropRateLR, nbEpochsDropLR, minimumLR)
				neuralLM.assignLearningRate(sess, newLR)

				# Training progress
				print "-> Epoch {}: Training Progress with Learning Rate {}".format(e, newLR)
				currStep = run_epoch(sess, neuralLM, trainData, batchSize, trainSummaryWriter, verbose=True, debug=True)

				# Development progress
				print "-> Epoch {}: Development Progress".format(e)
				cost, acc = evaluation_run(sess, currStep, neuralLM, devData, devSummaryWriter, verbose=True)

				# Save checkpoint
				path = saver.save(sess, ckpPrefix, global_step=currStep)
				print "-> Epoch {}: Saved Checkpoint - {}".format(e, path)

				# TODO: Early stopping

			# Save graph
			tf.train.write_graph(sess.graph, path2out, 'NeuralLM.pbtxt')







if __name__ == '__main__':
	main()