import io
import os
import yaml
import pickle
import operator
import regex as re
import numpy as np

NUM_SYM = u'<NUM>'
PUNC_SYM = u'<PUNC>'
RARE_SYM = u'<RARE>'
UNK_SYM = u'<UNK>'


def getFilesInDir(path2dir, fileExt):
	lstFile = []

	for root, dirs, files in os.walk(path2dir):
		for f in files:
			if f.endswith(fileExt):
				fileName = os.path.join(root, f)
				lstFile.append(fileName)

	return lstFile


def removeMultiUnderscore(word):
	return re.sub(ur"_+", ur"_", word, re.UNICODE)


def wordProcess(token):
	parts = token.split('/')

	wordParts = parts[:-2]
	word = '/'.join(wordParts)

	return removeMultiUnderscore(word)


def removeWordSegmentation(word):
	tokens = word.split('_')
	return tokens


def replacePUNC(lstToken):
	for idx in range(len(lstToken)):
		if lstToken[idx] in string.punctuation:
			lstToken[idx] = PUNC_SYM

	return lstToken


def replaceNUM(lstToken):
	numPattern = re.compile(ur'^\d[\d\P{l}]*', re.UNICODE)
	for idx in range(len(lstToken)):
		if numPattern.match(lstToken[idx]):
			lstToken[idx] = NUM_SYM
	return lstToken


def replaceRareTokens(lstToken, lstRareTokens):
	result = []
	for word in lstToken:
		if word in lstRareTokens:
			result.append(RARE_SYM)
		else:
			result.append(word)

	return result


def normalizeWord(lstToken):
	result = []
	for token in lstToken:
		token = re.sub(ur"([\p{l}+])\P{l}+$", ur"\1", token, re.UNICODE)
		result.append(token)

	return result


def eliminateRareTokenInDict(wordFreq, minFreq):
	newDict = wordFreq
	rareTokens = []
	nbRare = 0
	for k, v in wordFreq.items():
		if v < minFreq:
			del newDict[k]
			rareTokens.append(k)
			nbRare += v

	newDict[RARE_SYM] = nbRare

	return newDict, rareTokens


def getWordFreqDict(lstSents, minFreq):
	wordCounter = Counter()

	for sent in lstSents:
		wordCounter.update(sent)

	rawWordFreq = dict(wordCounter)

	return eliminateRareTokenInDict(rawWordFreq, minFreq)


def saveModelConfig(path2config, config):
	with open(path2config, 'w') as fout:
		yaml.dump(config, fout)


def loadModelConfig(path2config):
	config = {}
	with open(path2config) as fin:
		config = yaml.load(fin)
	return config


def loadData(path2dir, prefix, loadWordEmbedding=False):
	x = np.load(os.path.join(path2dir, '{}-input.npy'.format(prefix)))
	xLen = np.load(os.path.join(path2dir, '{}-inputLen.npy'.format(prefix)))
	y = np.load(os.path.join(path2dir, '{}-output.npy'.format(prefix)))
	if loadWordEmbedding:
		wordEmbedding = np.load(os.path.join(path2dir, 'word-embedding.npy'))
		return (x, xLen, y), wordEmbedding
	return (x, xLen, y)


def saveData(path2dir, prefix, wordEmbedding, input, inputLen, output):
	np.save(os.path.join(path2dir, '{}-input.npy'.format(prefix)), input)
	np.save(os.path.join(path2dir, '{}-inputLen.npy'.format(prefix)), inputLen)
	np.save(os.path.join(path2dir, '{}-output.npy'.format(prefix)), output)
	if wordEmbedding is not None:
		np.save(os.path.join(path2dir, 'word-embedding.npy'), wordEmbedding)


def loadPretrainWordEmbedding(path2pretrainWordEmbedding):
	wordEmbedding = {}
	with io.open(path2pretrainWordEmbedding, encoding='utf8') as fin:
		for line in fin:
			line = line.strip()
			values = line.split()
			word = values[0]
			weights = np.asarray(values[1:], dtype=np.float32)
			wordEmbedding[word] = weights
	return wordEmbedding


def loadPKLFile(path2file):
	fin = open(path2file, 'rb')
	result = pickle.load(fin)
	fin.close()
	return result


def savePKLFile(path2file, obj):
	with open(path2file, 'wb') as fout:
		pickle.dump(obj, fout)


def sortDictByValue(dInput, reverse=False):
	return sorted(dInput.items(), key=operator.itemgetter(1), reverse=reverse)


def unisonShuffleNumpyArrays(*arrays):
	# Check equal number of items in axis 0
	setLen = set()
	for arr in arrays:
		setLen.add(arr.shape[0])

	assert len(setLen) == 1

	pIndices = np.random.permutation(setLen.pop())

	results = []
	for arr in arrays:
		results.append(arr[pIndices])

	return results


def getStatsInfo(lstSents, wordFreqDict, rareTokens):
	print "Number of Sents:", len(lstSents)
	print "Number of Unique Token:", len(wordFreqDict)
	print "Number of Rare Token:", len(rareTokens)
	print "=============="

	top50Word = sortDictByValue(wordFreqDict, True)[:50]
	for word, freq in top50Word:
		print word, freq

	print "=============="
	bottom50Word = sortDictByValue(wordFreqDict)[:50]
	for word, freq in bottom50Word:
		print word, freq


	print "=============="
	lstSentLens = [len(sent) for sent in lstSents]
	sentLenDist = dict(Counter(lstSentLens))

	for sentLen, freq in sortDictByValue(sentLenDist, True):
		print sentLen, freq


def main():
	# a = np.array(range(10), dtype=np.int)
	# b = np.array(range(20, 30), dtype=np.int)

	# c, d = unisonShuffleNumpyArrays(a, b)
	# print a
	# print b
	# print c
	# print d

	path2config = "NeuralLM.config"
	config = loadModelConfig(path2config)

	print config



if __name__ == '__main__':
	main()