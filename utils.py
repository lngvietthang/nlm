import yaml

def saveModelConfig(path2config, config):
	with open(path2config, 'w') as fout:
		yaml.dump(config, fout)


def loadModelConfig(path2config):
	config = {}
	with open(path2config) as fin:
		config = yaml.load(fin)
	return config


def loadData(path2data):
	x, xLen, y = None, None, None
	return (x, xLen, y)


def loadVocabulary(path2vocab):
	return None


def loadPretrainWordEmbedding(path2wordEmbedding):
	return None