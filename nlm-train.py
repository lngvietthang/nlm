import tensorflow as tf



def run_epoch(sess, model, data, batchSize):
	x, xLen, y = data

	totalIns = x.shape[0]

	nbBatch = totalIns / batchSize + (totalIns % batchSize > 0)

	for i in range(nbBatch):
		


def main():
	path2data = ""


if __name__ == '__main__':
	main()