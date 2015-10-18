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
	def __init__(self):
		Parser.__init__(self)

	def train(self, sentences):
		for s in sentences:
			trees = s.get_labeled_trees()
			i = 0
			no_construction = False
			while ( len(trees) > 0 ):
				if i == len(trees) - 1:
					if no_construction == True:
						break;
					no_construction = True
					i = 0
				else:
					x = get_features(T, i)
					y = estimate_action(model, x)
					take_action(T, i ,y)
					if y!="SHIFT":
						no_construction = False
