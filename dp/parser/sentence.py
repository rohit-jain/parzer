import node

class Sentence(object):
	"""unlabeled sentence"""

	def __init__(self, words):
		# the list of words in the sentence
		self.words = words
		self.trees = []
		position = 0
		for word in words:
			self.trees += [ node.Node(word, position)]
			position += 1

	def get_trees():
		return self.trees

	def __repr__(self):
		words = str(self.words)
		pos_tags = str(self.pos_tags)
		dependency = str(self.dependency)
		trees = str(self.trees)
		return "<Sentence words: %s, trees: %s>" % (words, trees)


class ParsedSentence(Sentence):
	"""labeled sentence with the parsed positions"""

	def __init__(self, words, pos_tags, dependency):
		Sentence.__init__(self, words)
		self.pos_tags = pos_tags
		self.dependency = dependency
		self.trees = []
		for i in range(len(self.words)):
			self.trees += [ node.Node(self.words[i], i, self.pos_tags[i], self.dependency[i]) ]


	def get_labeled_trees(self):
		return self.trees

	def __repr__(self):
		words = str(self.words)
		pos_tags = str(self.pos_tags)
		dependency = str(self.dependency)
		trees = str(self.trees)
		return "<ParsedSentence words: %s, pos_tags: %s, dependencies: %s, trees: %s>" % (words, pos_tags, dependency, trees)