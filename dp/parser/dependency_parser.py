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
				return "RIGHT"
		elif b.dependency == a.position and self.complete_subtree(trees, b):
				return "LEFT"
		else:
			return "SHIFT"

	def take_action(self, trees, position, action):
		a = trees[position]
		b = trees[position + 1]
			
		if action == "RIGHT":
			b.insert_right(a)
			trees[position + 1] = b
			trees.remove(a)
		elif action == "LEFT":
			a.insert_left(b)
			trees[position] = a
			trees.remove(b)
		return trees

	def extract_features(self, trees, i):
		lex = dict.fromkeys(self.vocab, False)
		tags = dict.fromkeys(self.tags, False)
		for word in trees[i].lex:
			lex[word] = True
		for tag in trees[i].pos_tag:
			tags[tag] = True
		return lex.values() + tags.values()

	def train(self, sentences):
		train_x = []
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
					train_x += [self.extract_features(trees, i)]
					# estimate the action to be taken for i, i+ 1 target  nodes
					y = self.estimate_action(trees, i)
					# execute the action and modify the trees
					if y!="SHIFT":
						trees = self.take_action(trees, i ,y)
						no_construction = False
					else:
						i += 1
		print train_x