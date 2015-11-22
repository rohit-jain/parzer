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
		self.st = StanfordPOSTagger("wsj-0-18-bidirectional-distsim.tagger")
		self.clf = {}
		# pickle.load( open( "svm2.p", "rb" ) )
		self.actions = Counter()
		self.N_FEATURES = (3 * len(self.vocab)) + (3 * len(self.tags))
		

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
			# action_array = [SHIFT, LEFT, RIGHT]
			# random.shuffle(action_array)
			action_array = [i[0] for i in self.actions.most_common(1)]
			print action_array
		return action_array[0]


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

		lex_index = self.vocab[("<UNKNOWN>")]
		left_lex_index = len(self.vocab) + self.vocab[("<UNKNOWN>")]
		right_lex_index = 2*len(self.vocab) + self.vocab[("<UNKNOWN>")]

		tag_index = 3*len(self.vocab) + self.tags[("<UNKNOWN>")]
		left_tag_index = 3*len(self.vocab) + len(self.tags) + self.tags[("<UNKNOWN>")]
		right_tag_index = 3*len(self.vocab) + 2*len(self.tags) + self.tags[("<UNKNOWN>")]
		
		if ((target_node.lex) in self.vocab):
			lex_index = self.vocab[(target_node.lex)]

		if ( target_node.pos_tag in self.tags):
			tag_index = 3*len(self.vocab) + self.tags[(target_node.pos_tag)]


		if( i!= 0 ):
			left_target_node = trees[i-1]
			if ((left_target_node.lex) in self.vocab):
				left_lex_index = len(self.vocab) + self.vocab[(left_target_node.lex)]
			if ( left_target_node.pos_tag in self.tags):
				left_tag_index = 3*len(self.vocab) + len(self.tags) + self.tags[(left_target_node.pos_tag)]

		if( i < (len(trees) - 1) ):
			right_target_node = trees[i+1]
			if ((right_target_node.lex) in self.vocab):
				right_lex_index = 2*len(self.vocab) + self.vocab[(right_target_node.lex)]
			if ( right_target_node.pos_tag in self.tags):
				right_tag_index = 3*len(self.vocab) + 2*len(self.tags) + self.tags[(right_target_node.pos_tag)]
			

		return [lex_index, tag_index, left_lex_index, left_tag_index, right_lex_index, right_tag_index]

	def get_pos(self, trees, i):
		target_node = trees[i]
		return target_node.pos_tag

	def train(self, sentences):
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
					extracted_features = self.extract_features(trees, i)
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

					# execute the action and modify the trees
					if y!= SHIFT:
						trees = self.take_action(trees, i ,y)
						no_construction = False
					else:
						i += 1

		for lp in train_x:
			print lp
			print len(train_x[lp])
			temp_features = lil_matrix((len(train_x[lp]), self.N_FEATURES), dtype = bool)
			print self.N_FEATURES

			for i in range( 0, len(train_x[lp]) ):
				lex_index, tag_index, left_lex_index, left_tag_index, right_lex_index, right_tag_index = train_x[lp][i]
				temp_features[ i,lex_index ], temp_features[ i,tag_index ] = True, True
				temp_features[ i,left_lex_index ], temp_features[ i,left_tag_index ] = True, True
				temp_features[ i,right_lex_index ], temp_features[ i,right_tag_index ] = True, True

			features[lp] = temp_features.tocsr()

			train_x[lp] = None
			n_classes = Set()
			for i in train_y[lp]:
				n_classes.add(i)
			if( len(n_classes) > 1 ):
				clf[lp] = svm.SVC(kernel='poly', degree=2)
				clf[lp].fit(features[lp], train_y[lp])

		self.clf = clf
		print "pickling"
		pickle.dump( clf , open( "svm2.p", "wb" ) )
		print "pickling done"

	def test(self, sentences):
		test_sentences = []
		inferred_trees = []
		for s in sentences:
			test_sentences += [sentence.Sentence( s.words, s.pos_tags )]

		for s in test_sentences:
			trees = s.get_trees()
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
					extracted_features = self.extract_features(trees, i)

					# estimate the action to be taken for i, i+ 1 target  nodes
					y = self.estimate_action(trees, i, extracted_features)
					# execute the action and modify the trees
					if y!= SHIFT:
						trees = self.take_action(trees, i ,y)
						no_construction = False
					else:
						i += 1
			inferred_trees += [trees]
		return inferred_trees


	def evaluate(self, inferred_trees, gold_sentences):
		correct_roots = 0
		correct_parents = 0
		total_parents = 0
		total_sentences = len(gold_sentences)
		complete_parses = 0

		for i,it in enumerate(inferred_trees):
			if(len(it) == 1):
				complete_parses += 1
				s = gold_sentences[i]
				# print s.words
				# print s.dependency
				# print it[0]
				correct_parents += it[0].match(s)
				total_parents += (len(s.words) - 1)

				if( s.dependency[it[0].position] == -1 ):
					correct_roots += 1

		print correct_roots
		print correct_parents
		print complete_parses
		print total_sentences
		print "root accuracy: " + str(correct_roots/total_sentences)
		print "dependency accuracy: " + str(correct_parents/total_parents)
		print "completion rate: " + str(complete_parses/total_sentences)