import logging
import os
import sentence
import dependency_parser

LOG_FILENAME = 'logging.out'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)

def read_penn_treebank( path, low, high):
	"""
	Read sentences from dependency 
	converted penn treebank files
	"""
	sentences = []
	s = int(low[0:2])
	e = int(high[0:2])
	sentences = []

	while( s <= e ):
		segment = path + str(s).zfill(2)
		for f in os.listdir(segment):
			file_path = segment + "/" + f
			if (not f.startswith('.')) and os.path.isfile( file_path ) and (int(f[-4:]) <= int(high)):
				sentence_file = open( file_path , "r") 
				words = []
				pos_tags = []
				dependencies = []
				for line in open( file_path , "r"):
					if (line != "\n"):
						word, tag, dependency = line.strip("\n").split("\t")
						words += [ word ]
						pos_tags += [ tag ]
						dependencies += [ int(dependency) ]
					else:
						# test with only one sentence
						# if len(sentences) == 1:
						# 	break
						# if len(words) >= 6 and len(words) <= 14:
						sentences += [sentence.ParsedSentence( words, pos_tags, dependencies )]
						words = []
						pos_tags = []
						dependencies = []
		s += 1
	return sentences

if __name__ == '__main__':
	DATA_PATH = "/Users/rohitjain/github/nlp/dp/data/wsj_parsed/"
	# Read train sentences from penn treebank for the given sections with labels
	logging.info("Reading training data")
	training_sentences = read_penn_treebank(DATA_PATH, "0001", "0010")
	# Read validate sentences from penn treebank for the given sections without labels
	valdation_sentences = read_penn_treebank(DATA_PATH, "0011", "0021")
	# Initialise parser
	my_parser = dependency_parser.SVMParser()
	# train the data
	my_parser.train( training_sentences )