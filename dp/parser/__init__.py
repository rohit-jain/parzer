import logging

class Sentence(object):
	"""Sentence from the penn treebank"""

	def __init__(self, arg):
		super(Sentence, self).__init__()
		# the list of words in the sentence
		self.words = []
		self.positions = []


class ParsedSentence(Sentence):
	"""sentence with the parsed positions"""

	def __init__(self, arg):
		super(ParsedSentence, self).__init__()
		self.dependency = []
		self.arg = arg
		
if __name__ == '__main__':
	# Read train sentences from penn treebank for the given sections with labels
	logging.info("Reading training data")
	# Read validate sentences from penn treebank for the given sections without labels