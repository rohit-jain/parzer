class Node(object):
	"""
	Node of a tree
	Can contain any number of children
	left and right represents the children created due to left and right dependencies respectively
	"""
	def __init__(self, lex = None, position = None, pos_tag = None, dependency = None, left = [], right = []):
		self.lex = lex
		self.position = position
		self.pos_tag = pos_tag
		self.dependency = dependency
		self.left = left
		self.right = right

	def insert_right(self, child):
		self.right = self.right + [child]

	def insert_left(self, child):
		self.left = self.left + [child]

	def __str__(self):
		temp_left = []
		temp_right = []
		# print len(self.right)
		if len(self.right) > 0:
			for r in self.right:
				temp_right += [str(r)]

		if len(self.left) > 0:
			for l in self.left:
				temp_left += [str(l)]
		return "lex: %s, pos_tag: %s, left: %s, right: %s\n" % (self.lex, self.pos_tag, "[" + "\n".join(temp_left) + "]", "[" + "\n".join(temp_right) + "]")
	
	def __repr__(self):
		temp_left = []
		temp_right = []
		# print len(self.right)
		if len(self.right) > 0:
			for r in self.right:
				temp_right += [str(r)]

		if len(self.left) > 0:
			for l in self.left:
				temp_left += [str(l)]
		return "<Node lex: %s, position: %s, pos_tag: %s, dependency: %s, left: %s, right: %s>\n" % (self.lex, self.position, self.pos_tag, self.dependency, "[" + "\n".join(temp_left) + "]", "[" + "\n".join(temp_right) + "]")