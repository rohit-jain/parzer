import logging
import os

LOG_FILENAME = 'logging.out'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)

class Parser(object):
	"""
	Abstract Class ( Interface ) for Parser
	"""
	def __init__(self, arg):
		self.arg = arg
	
	def train():
        raise NotImplementedError( "Should have implemented this" )

    def validate():
	    raise NotImplementedError( "Should have implemented this" )

	def test():
        raise NotImplementedError( "Should have implemented this" )		

class SVMParser(Parser):
	"""
	Dependency parser based on yamada et al ( 2003 )
	"""
	def __init__(self, arg):
		Parser.__init__(self)
		self.arg = arg
		

class Sentence(object):
	"""unlabeled sentence"""

	def __init__(self, words):
		# the list of words in the sentence
		self.words = words


class ParsedSentence(Sentence):
	"""labeled sentence with the parsed positions"""

	def __init__(self, words, pos_tags, dependency):
		Sentence.__init__(self, words)
		self.pos_tags = pos_tags
		self.dependency = dependency

	def __repr__(self):
		words = str(self.words)
		pos_tags = str(self.pos_tags)
		dependency = str(self.dependency)
		return "<ParsedSentence words: %s, pos_tags: %s, dependencies: %s>" % (words, pos_tags, dependency)

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
						sentences += [ParsedSentence( words, pos_tags, dependencies )]
		s += 1
	return sentences

if __name__ == '__main__':
	DATA_PATH = "/Users/rohitjain/github/nlp/dp/data/wsj_parsed/"
	# Read train sentences from penn treebank for the given sections with labels
	logging.info("Reading training data")
	training_sentences = read_penn_treebank(DATA_PATH, "0001", "0011")
	# Read validate sentences from penn treebank for the given sections without labels
	valdation_sentences = read_penn_treebank(DATA_PATH, "0011", "0021")
	# Initialise parser
	# train the data