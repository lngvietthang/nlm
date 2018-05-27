# -*- coding: utf-8 -*-
# @Author: cpu11437-local
# @Date:   2018-05-03 09:45:24
# @Last Modified by:   vietthang
# @Last Modified time: 2018-05-27 20:20:38

import io
import os
import pickle 

import utils

def getWord(lstToken):
	lstWords = []
	for token in lstToken:
		lstWords.append(utils.wordProcess(token))

	return lstWords


def readData(path2data):
	lstFiles = utils.getFilesInDir(path2data, 'txt')
	lstSents = []
	for file in lstFiles:
		with io.open(file, encoding='utf16') as fin:
			for line in fin:
				line = line.strip()
				if '' == line:
					continue

				lstToken = line.split(' ')
				lstSents.append(getWord(lstToken))
	return lstSents


def preprocess(lstSents):
	lstPPSents = []
	for sent in lstSents:
		ppSent = [u'<SS>']
		for word in sent:
			word = word.lower()
			subWords = utils.removeWordSegmentation(word)
			subWords = utils.normalizeWord(subWords)
			subWords = utils.replacePUNC(subWords)
			subWords = utils.replaceNUM(subWords)
			ppSent += subWords
		lstPPSents.append(ppSent)

	return lstPPSents



def normalizeSentence(lstSents, rareTokens):
	lstNormSents = []
	for sent in lstSents:
		normSent = utils.replaceRareTokens(sent, rareTokens)
		lstNormSents.append(normSent)

	return lstNormSents


def export(path2out, data, wordFreqDict, rareTokens):
	path2wordFreq = os.path.join(path2out, 'wordfreq.pkl')
	path2rareTokens = os.path.join(path2out, 'raretoken.pkl')
	path2data = os.path.join(path2out, 'data.pkl')

	with io.open(path2wordFreq, 'wb') as fout:
		pickle.dump(wordFreqDict, fout)

	with io.open(path2rareTokens, 'wb') as fout:
		pickle.dump(rareTokens, fout)

	with io.open(path2data, 'wb') as fout:
		pickle.dump(data, fout)


def exportDatasetInTextFormat(path2out, data):
	with io.open(path2out, 'w', encoding='utf8') as fout:
		for line in data:
			sent = u" ".join(line)
			fout.write(sent + u"\n")
			fout.flush()


def main():
	path2data = 'data/raw/'
	path2out = 'data/post-process'
	minFreq = 3

	lstSents = readData(path2data)
	# print ur" ".join(lstSents[0])
	lstSents = preprocess(lstSents)

	wordFreqDict, rareTokens = utils.getVocabulary(lstSents, minFreq)

	lstSents = normalizeSentence(lstSents, rareTokens)

	export(path2out, lstSents, wordFreqDict, rareTokens)

	# exportDatasetInTextFormat(os.path.join(path2out, 'wordembedding-training-data.txt'), lstSents)


	# import random
	# nbSents = len(lstSents)
	# for i in range(10):
	# 	print u" ".join(lstSents[random.randint(0, nbSents-1)])

	utils.getStatsInfo(lstSents, wordFreqDict, rareTokens)


	return False


if __name__ == '__main__':
	main()