import os
import yaml
import pickle
import numpy as np

def saveModelConfig(path2config, config):
	with open(path2config, 'w') as fout:
		yaml.dump(config, fout)


def loadModelConfig(path2config):
	config = {}
	with open(path2config) as fin:
		config = yaml.load(fin)
	return config


def loadData(path2dir, prefix):
	x = np.load(os.path.join(path2dir, '{}-input.npy'.format(prefix)))
	xLen = np.load(os.path.join(path2dir, '{}-inputLen.npy'.format(prefix)))
	y = np.load(os.path.join(path2dir, '{}-output.npy'.format(prefix)))
	return (x, xLen, y)


def saveData(path2dir, prefix, input, inputLen, output):
	np.save(os.path.join(path2dir, '{}-input.npy'.format(prefix)), input)
	np.save(os.path.join(path2dir, '{}-inputLen.npy'.format(prefix)), inputLen)
	np.save(os.path.join(path2dir, '{}-output.npy'.format(prefix)), output)


def loadPretrainWordEmbedding(path2wordEmbedding):
	return None


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


def main():
	a = np.array(range(10), dtype=np.int)
	b = np.array(range(20, 30), dtype=np.int)

	c, d = unisonShuffleNumpyArrays(a, b)
	print a
	print b
	print c
	print d


if __name__ == '__main__':
	main()