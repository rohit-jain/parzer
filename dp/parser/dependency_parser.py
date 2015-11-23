from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn import svm
from nltk.tag import StanfordPOSTagger
import sentence
import pickle, random
from sets import Set
from collections import Counter

LEFT = 0
SHIFT = 1
RIGHT = 2
LEFT_CONTEXT = 2
RIGHT_CONTEXT = 4

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
	def __init__(self, vocab, tags, load=False):
		Parser.__init__(self)
		self.vocab = vocab
		self.tags = tags
		self.st = StanfordPOSTagger("wsj-0-18-bidirectional-distsim.tagger")
		self.clf = {}
		self.loaded = False
		if load == True:
			self.loaded = True
			self.clf = pickle.load( open( "svm_focus.p", "rb" ) )
		self.actions = Counter()
		self.target_feature_size = (3 * len(self.vocab)) + (3 * len(self.tags))
		self.context_feature_size = ( len(self.vocab) + len(self.tags) )
		self.N_FEATURES =  (LEFT_CONTEXT + RIGHT_CONTEXT) * self.context_feature_size + 2 * self.target_feature_size
		

	def complete_subtree(self, trees, child):
		for t in trees:
			if t.dependency == child.position:
				return False
		return True

	def estimate_train_action(self, trees, position):
		a = trees[position]
		b = trees[position + 1]
		if a.dependency == b.position and self.complete_subtree(trees, a):
				return RIGHT
		elif b.dependency == a.position and self.complete_subtree(trees, b):
				return LEFT
		else:
			return SHIFT


	def estimate_action(self, trees, position, extracted_features):
		tree_pos_tag = self.get_pos( trees, position )
		temp_features = lil_matrix((1,self.N_FEATURES), dtype = bool)
		for i in extracted_features:
			temp_features[0,i] = True
		if tree_pos_tag in self.clf:
			action_array = self.clf[tree_pos_tag].predict( temp_features )
		else:
			print "guess"
			action_array = [SHIFT, LEFT, RIGHT]
		return action_array[0]


	def take_action(self, trees, position, action):
		a = trees[position]
		b = trees[position + 1]
			
		if action == RIGHT:
			b.insert_right(a)
			trees[position + 1] = b
			trees.remove(a)
			if position == 0:
				position = 1
			return position-1, trees

		elif action == LEFT:
			a.insert_left(b)
			trees[position] = a
			trees.remove(b)
			if position == 0:
				position = 1

			return position-1, trees
		return position+1, trees

	def lex_feature( self, node, offset ):
		lex_index = self.vocab[("<UNKNOWN>")]
		if ((node.lex) in self.vocab):
			lex_index = self.vocab[(node.lex)]
		return lex_index + offset

	def pos_feature( self, node, offset ):
		tag_index = self.tags[("<UNKNOWN>")]
		if ( node.pos_tag in self.tags):
			tag_index = self.tags[(node.pos_tag)]
		return tag_index + offset

	def child_lex( self, children, offset ):
		lex_indices = []
		for child in children:
			lex_indices += [self.lex_feature( child,offset )]
		return lex_indices

	def child_pos( self, children, offset ):
		pos_indices = []
		for child in children:
			pos_indices += [self.pos_feature( child,offset )]
		return pos_indices


	def node_features( self, target_node, offset, child_features=False ):
		v = len(self.vocab)
		t = len(self.tags)
		lex = [self.lex_feature( target_node,0 ) + offset]
		pos = [self.pos_feature( target_node,v ) + offset]
		if child_features == True:
			ch_l_lex = [i+offset for i in self.child_lex( target_node.left,v+t ) ]
			ch_l_pos = [i+offset for i in self.child_pos( target_node.left,2*v+t ) ]
			ch_r_lex = [i+offset for i in self.child_lex( target_node.right,(2*v)+(2*t) ) ]
			ch_r_pos = [i+offset for i in self.child_pos( target_node.right,(3*v)+(2*t) ) ]
			return lex + pos + ch_l_lex + ch_l_pos + ch_r_lex + ch_r_pos
		return lex + pos


	def extract_features(self, trees, i, l, r):
		features = []
		for k,w in enumerate(range(i-l,(i+1+r+1))):
			if( w>= 0) and ( w< len(trees)):
				target_node = trees[w]
				if ( k<l ):
					features += self.node_features( target_node,k*self.context_feature_size )
				elif ( k>=l and k<l+2 ):
					target_offset = 2*self.context_feature_size + (k-l)*self.target_feature_size
					features += self.node_features( target_node,target_offset,child_features=True )
				else:
					target_offset = (2 + k - (l+2) )*self.context_feature_size + 2*self.target_feature_size
					features += self.node_features( target_node,target_offset )
		
		return features

	def get_pos(self, trees, i):
		target_node = trees[i]
		return target_node.pos_tag

	def train(self, sentences):
		if(self.loaded):
			return
		m = len(sentences)
		print m
		train_x = {}
		train_y = {}
		features = {}
		clf = {}

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
					tree_pos_tag = self.get_pos(trees, i)
					
					# extract features
					extracted_features = self.extract_features(trees, i, LEFT_CONTEXT, RIGHT_CONTEXT)
					# print (extracted_features)

					# estimate the action to be taken for i, i+ 1 target  nodes
					y = self.estimate_train_action(trees, i)
					self.actions[y] += 1

					if tree_pos_tag in train_x:
						train_x[tree_pos_tag].append( extracted_features )
						train_y[tree_pos_tag].append( y )

					else:
						train_x[tree_pos_tag] = [extracted_features]
						train_y[tree_pos_tag] = [y]

					i, trees = self.take_action(trees, i ,y)

					# execute the action and modify the trees
					if y!= SHIFT:
						no_construction = False

		for lp in train_x:
			print lp
			print len(train_x[lp])
			temp_features = lil_matrix((len(train_x[lp]), self.N_FEATURES), dtype = bool)
			print self.N_FEATURES

			for i in range( 0, len(train_x[lp]) ):
				for k in train_x[lp][i]:
					temp_features[ i,k ] = True

			features[lp] = temp_features.tocsr()

			train_x[lp] = None
			n_classes = Set()
			for i in train_y[lp]:
				n_classes.add(i)
			if( len(n_classes) > 1 ):
				clf[lp] = svm.SVC(kernel='poly', degree=2)
				# clf[lp] = svm.LinearSVC()
				clf[lp].fit(features[lp], train_y[lp])

		self.clf = clf
		print "pickling"
		pickle.dump( clf , open( "svm_focus.p", "wb" ) )
		print "pickling done"

	def test(self, sentences):
		test_sentences = []
		inferred_trees = []
		total = 0
		for s in sentences:
			test_sentences += [sentence.Sentence( s.words, s.pos_tags )]

		for s in test_sentences:
			trees = s.get_trees()
			i = 0
			no_construction = False
			while ( len(trees) > 0 ):
				if i == (len(trees) - 1):
					if no_construction == True:
						break;
					# if we reach the end start from the beginning
					no_construction = True
					i = 0
				else:					
					# extract features
					extracted_features = self.extract_features(trees, i, LEFT_CONTEXT, RIGHT_CONTEXT)

					# estimate the action to be taken for i, i+ 1 target  nodes
					y = self.estimate_action(trees, i, extracted_features)
					i, trees = self.take_action(trees, i ,y)
					# execute the action and modify the trees
					if y!= SHIFT:
						no_construction = False
			if(len(trees) == 1):
				total+=1
			inferred_trees += [trees]
		print total
		return inferred_trees


	def evaluate(self, inferred_trees, gold_sentences):
		correct_roots = 0
		correct_parents = 0
		total_parents = 0
		total_sentences = len(gold_sentences)
		complete_parses = 0

		for i,it in enumerate(inferred_trees):
			s = gold_sentences[i]
			total_parents += (len(s.words) - 1)

			if(len(it) == 1):
				if( s.dependency[it[0].position] == -1 ):
					correct_roots += 1
				if it[0].match_all(s):
					complete_parses += 1


			for t in it:
				correct_parents += t.match(s)


		print correct_roots
		print correct_parents
		print total_parents
		print complete_parses
		print total_sentences
		print "root accuracy: " + str(correct_roots/total_sentences)
		print "dependency accuracy: " + str(correct_parents/total_parents)
		print "completion rate: " + str(complete_parses/total_sentences)