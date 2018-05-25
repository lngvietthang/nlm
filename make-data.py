# -*- coding: utf-8 -*-
# @Author: vietthang
# @Date:   2018-05-25 22:53:32
# @Last Modified by:   vietthang
# @Last Modified time: 2018-05-26 00:17:09
import os
import numpy as np

import utils


def buildVocabIndex(wordFreqDict, indexStart = 1):
	# indexStart = 1 <- reverse 0 for padding symbol
	vocabIndex = {UNK_SYM : indexStart}
	sortedVocab = utils.sortDictByValue(wordFreqDict, True)
	idx = indexStart + 1
	for word, _ in sortedVocab:
		vocabIndex[word] = idx
		idx += 1

	invVocabIndex = {idx: word for word, idx in vocabIndex.items()}
	return vocabIndex, invVocabIndex


def getSegment(sent, maxToken):
	result = []
	for i in range(1, len(sent)):
		start = max(0, i - maxToken + 1)
		end = i + 1
		result.append(sent[start:end])

	return result


def mapWord2Index(lstWord, vocabIndex):
	result = []
	for word in lstWord:
		if word in vocabIndex:
			result.append(vocabIndex[word])
		else:
			result.append(vocabIndex[UNK_SYM])

	return result


def makeNeuralLMDataFormat(data, vocab, contextLen = 6):
	input, inputLen, output = [], [], []
	for sent in data:
		lstSegment = getSegment(sent, contextLen + 1)
		for seg in lstSegment:
			seg = mapWord2Index(seg, vocab)
			i = seg[:-1]
			iL = len(i)
			o = seg[-1]
			if iL < contextLen:
				paddingLen = contextLen - iL
				i += [0] * paddingLen
			input.append(i)
			inputLen.append(iL)
			output.append(o)
			

	return input, inputLen, output


def convertList2NumpyArray(lst):
	return np.array(lst, dtype=np.int)


def main():
	path2data = ""
	path2wordFreqDict = ""
	devRatio = 0.1

	path2dirOut = ""

	data = utils.loadPKLFile(path2data)
	wordFreqDict = utils.loadPKLFile(path2wordFreqDict)

	vocab, invertVocab = buildVocabIndex(wordFreqDict)

	input, inputLen, output = makeNeuralLMDataFormat(data, vocab)

	input = convertList2NumpyArray(input)
	inputLen = convertList2NumpyArray(inputLen)
	output = convertList2NumpyArray(output)

	# Split
	input, inputLen, output = utils.unisonShuffleNumpyArrays(input, inputLen, output)

	totalIns = input.shape[0]

	nbDev = int(totalIns * devRatio)

	train = (input[:-nbDev], inputLen[:-nbDev], output[:-nbDev])
	dev = (input[-nbDev:], inputLen[-nbDev:], output[-nbDev:])


	utils.saveData(path2dirOut, 'train', *train)
	utils.saveData(path2dirOut, 'dev', *dev)
	
	utils.savePKLFile(os.path.join(path2dirOut, 'vocab.pkl'), vocab)
	utils.savePKLFile(os.path.join(path2dirOut, 'invert-vocab.pkl'), invertVocab)


if __name__ == '__main__':
	main()