import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn import svm

LEFT = 0
SHIFT = 1
RIGHT = 2

class Parser(object):
	"""
	Abstract Class ( Interface ) for Parser
	"""
	def __init__(self):
		pass
	
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
	def __init__(self, vocab, tags):
		Parser.__init__(self)
		self.vocab = vocab
		self.tags = tags

	def complete_subtree(self, trees, child):
		for t in trees:
			if t.dependency == child.position:
				return False
		return True

	def estimate_action(self, trees, position):
		a = trees[position]
		b = trees[position + 1]
		if a.dependency == b.position and self.complete_subtree(trees, a):
				return RIGHT
		elif b.dependency == a.position and self.complete_subtree(trees, b):
				return LEFT
		else:
			return SHIFT

	def take_action(self, trees, position, action):
		a = trees[position]
		b = trees[position + 1]
			
		if action == RIGHT:
			b.insert_right(a)
			trees[position + 1] = b
			trees.remove(a)
		elif action == LEFT:
			a.insert_left(b)
			trees[position] = a
			trees.remove(b)
		return trees

	def extract_features(self, trees, i):
		target_node = trees[i]
		lex_index = self.vocab[(target_node.lex)]
		tag_index = len(self.vocab) + self.tags[(target_node.pos_tag)]
		return [lex_index, tag_index]

	def train(self, sentences):
		m = len(sentences)
		n = len(self.vocab) + len(self.tags)
		print m
		print n
		# train_x = lil_matrix((m, n), dtype=np.bool)
		train_x = []
		train_y = []
		for s in sentences:
			trees = s.get_labeled_trees()
			# print "Original"
			# print s.trees
			i = 0
			no_construction = False
			while ( len(trees) > 0 ):
				if i == len(trees) - 1:
					if no_construction == True:
						break;
					# if we reach the end start from the beginning
					no_construction = True
					i = 0
				else:
					# extract features
					train_x.append(self.extract_features(trees, i))
					# estimate the action to be taken for i, i+ 1 target  nodes
					y = self.estimate_action(trees, i)
					train_y += [y]
					# execute the action and modify the trees
					if y!= SHIFT:
						trees = self.take_action(trees, i ,y)
						no_construction = False
					else:
						i += 1
		features = lil_matrix((len(train_x),n), dtype = bool)

		for i,j in train_x:
			features[i,j] = True
		features = features.tocsr()
		print len(train_x)
		# features = np.zeros((len(train_x),n), dtype = bool)
		# for i,j in train_x:
		# 	features[i][j] = True

		train_x = None
		clf = svm.LinearSVC()
		clf.fit(features, train_y)
		# print features