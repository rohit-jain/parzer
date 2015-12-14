from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn import svm
import sentence
import pickle, random
from sets import Set
from collections import Counter
import os
from sklearn.grid_search import GridSearchCV
import getopt
import copy

LEFT = 0
SHIFT = 1
RIGHT = 2
# length of the left context window
LEFT_CONTEXT = 2
# length of the right context window
RIGHT_CONTEXT = 4
FINAL_MODEL = "svm"
EXTRACT_LEX = 0
EXTRACT_POS = 1

def counter_ratio(n,d):
    r = dict()
    for i in n:
        r[i] = n[i]/d[i]
    return r

def convert_to_features(d):
    """
    Convert vocab counters for each positon to one hot feature vectors
    For each position there is a dictionary with each word mapped to a specific index 
    """
    tpt = {}
    for pi in d:
        tpt[pi] = {}
        i = 0
        for token in d[pi]:
            tpt[pi][token] = i
            i += 1
        tpt[pi]["<UNKNOWN>"] = i
    return tpt

def count_features(d):
    all_d = 0
    for pi in d:
        l = len(d[pi])
        # print pi,l
        all_d += l
    return all_d


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
    def __init__(self, model, load=False, cache_size=5120):
        Parser.__init__(self)
        self.st = {}#StanfordPOSTagger("wsj-0-18-bidirectional-distsim.tagger")
        self.clf = {}
        self.position_vocab = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.position_tag = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.ch_l_tag = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.ch_r_tag = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.ch_l_vocab = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.ch_r_vocab = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.last_action = {}
        self.cache_size = cache_size
        self.loaded = False
        self.actions = Counter()
        self.test_actions = Counter()
        self.tag_actions = {}
        self.N_FEATURES = None
        self.model_path = model
        if load == True:
            self.loaded = True
            self.clf = pickle.load( open( self.model_path + FINAL_MODEL+".p", "rb" ) )
        

    def complete_subtree(self, trees, child):
        """
        Method to check if the tree is a complete subtree
        """
        for t in trees:
            if t.dependency == child.position:
                return False
        return True

    def estimate_train_action(self, trees, position):
        """
        Method to estimate parsing action for the training data
        """
        a = trees[position]
        b = trees[position + 1]
        if a.dependency == b.position and self.complete_subtree(trees, a):
                return RIGHT
        elif b.dependency == a.position and self.complete_subtree(trees, b):
                return LEFT
        else:
            return SHIFT


    def estimate_action(self, trees, position, extracted_features):
        """
        Method to infer parsing actions using the learned classifier
        """
        tree_pos_tag = self.get_pos( trees, position )
        temp_features = lil_matrix((1,self.N_FEATURES), dtype = bool)
        for i in extracted_features:
            temp_features[0,i] = True
        if tree_pos_tag in self.clf:
            try:
                action_array = self.clf[tree_pos_tag].predict( temp_features )
            except Exception as e:
                return self.tag_actions[tree_pos_tag].most_common(1)[0][0]
        else:
            action_array = [SHIFT, LEFT, RIGHT]
        return action_array[0]


    def take_action(self, trees, position, action):
        """
        Execute the parsing action given the trees and position
        Returns the new position of the target node and the transformed trees
        """
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

    def lex_feature( self, position, node, offset ):
        """
        Given a node and the position extract lex feature value using the vocabulary for that position
        """
        vocab = self.position_vocab[position]
        lex_index = vocab[("<UNKNOWN>")]
        if (node!=[]):
            if ((node.lex) in vocab):
                lex_index = vocab[(node.lex)]
        return lex_index + offset

    def pos_feature( self, position, node, offset ):
        """
        Given a node and the position extract pos tag feature value using the vocabulary for that position
        """
        tag = self.position_tag[position]
        tag_index = tag[("<UNKNOWN>")]
        if (node!=[]):
            if ( node.pos_tag in tag):
                tag_index = tag[(node.pos_tag)]
        return tag_index + offset

    def child_feature( self, position, node, offset, family, type_feature ):
        """
        extract features for given child node and position from the family(vocabulary)
        based on the type_feature value
        """
        vocab = family
        index = vocab[("<UNKNOWN>")]
        if (node!=[]):
            if(type_feature == EXTRACT_LEX):
                if ((node.lex) in vocab):
                    index = vocab[(node.lex)]
            else:
                if ((node.pos_tag) in vocab):
                    index = vocab[(node.pos_tag)]
        return index + offset

    def child_features( self, position, children, offset, family, type_feature ):
        """
        extract features for all given children and position from the family(vocabulary)
        """
        indices = []
        for child in children:
            indices += [self.child_feature( position,child,offset,family,type_feature )]
        return indices

    def extract_test_features(self, trees, i, l, r, total_offset):
        """
        Method to extract features for the given context window
        """
        features = []
        offset = 0
        for k,w in enumerate(range(i-l,(i+1+r+1))):
            if( w>= 0) and ( w< len(trees)):
                target_node = trees[w]
                temp_lex = [self.lex_feature(k, target_node, offset)]
                offset += len(self.position_vocab[k])
                temp_tag = [self.pos_feature(k, target_node, offset)]
                offset += len(self.position_tag[k])
                temp_ch_l_lex = self.child_features(k, target_node.left, offset, self.ch_l_vocab[k], EXTRACT_LEX)
                offset += len(self.ch_l_vocab[k])
                temp_ch_l_tag = self.child_features(k, target_node.left, offset, self.ch_l_tag[k], EXTRACT_POS)
                offset += len(self.ch_l_tag[k])
                temp_ch_r_lex = self.child_features(k, target_node.right, offset, self.ch_r_vocab[k], EXTRACT_LEX)
                offset += len(self.ch_r_vocab[k])
                temp_ch_r_tag = self.child_features(k, target_node.right, offset, self.ch_r_tag[k], EXTRACT_POS)
                offset += len(self.ch_r_tag[k])

                features += (temp_lex + temp_tag + temp_ch_r_tag + temp_ch_r_lex + temp_ch_l_tag + temp_ch_l_lex )

        return features


    def build_vocab(self, trees, i, l, r):
        """
        Build vocabulary counter for every position in the context window
        Input:
            trees: all the node trees for the sentence
            i: the position of left target tree(node)
            l: length of the left context window
            r: length of the right context window 
        """
        for k,w in enumerate(range(i-l,(i+1+r+1))):
            if( w>= 0) and ( w< len(trees)):
                target_node = trees[w]
                self.position_vocab[k][target_node.lex] += 1
                self.position_tag[k][target_node.pos_tag] += 1
                for lc in target_node.left:
                    self.ch_l_vocab[k][lc.lex] += 1
                    self.ch_l_tag[k][lc.pos_tag] += 1
                for rc in target_node.right:
                    self.ch_r_vocab[k][rc.lex] += 1
                    self.ch_r_tag[k][rc.pos_tag] += 1

    def get_pos(self, trees, i):
        target_node = trees[i]
        return target_node.pos_tag

    def update_pos_action(self, tree_tag, action):
        """ keep track of each action for every POS tag """
        if tree_tag not in self.tag_actions.keys():
            self.tag_actions[tree_tag] = Counter()
        self.tag_actions[tree_tag][action] += 1
        return

    def train(self, sentences):
        """
        Method to train the parser
        """
        sentences2 = copy.deepcopy(sentences)
        
        # go through the training sentences to build vocabulary for each position in the tree
        # construct the tree from data and learn the vocab for each position at each step
        for s in sentences:
            trees = s.get_labeled_trees()
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
                    
                    # build_vocab
                    self.build_vocab(trees, i, LEFT_CONTEXT, RIGHT_CONTEXT)

                    # estimate the action to be taken for i, i+ 1 target  nodes
                    y = self.estimate_train_action(trees, i)
                    i, trees = self.take_action(trees, i ,y)

                    # execute the action and modify the trees
                    if y!= SHIFT:
                        no_construction = False

        # convert vocabulary counters to one hot features
        self.position_tag = convert_to_features(self.position_tag)
        self.position_vocab = convert_to_features(self.position_vocab)
        self.ch_l_tag = convert_to_features(self.ch_l_tag)
        self.ch_r_tag = convert_to_features(self.ch_r_tag)
        self.ch_l_vocab = convert_to_features(self.ch_l_vocab)
        self.ch_r_vocab = convert_to_features(self.ch_r_vocab)

        # set the total number of features
        self.N_FEATURES = count_features(self.position_tag) + count_features(self.position_vocab) + count_features(self.ch_l_tag) + count_features(self.ch_l_vocab) + count_features(self.ch_r_tag) + count_features(self.ch_r_vocab)
        print self.N_FEATURES
        # if the model was loaded from disk
        # skip training and return
        if(self.loaded):
            return

        train_x = {}
        train_y = {}
        features = {}
        clf = {}

        # parsing through training sentences to build vectors for learning
        for s in sentences2:
            # get all labeled trees
            # all word nodes are independent trees in the begining
            trees = s.get_labeled_trees()
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
                    extracted_features = self.extract_test_features(trees, i, LEFT_CONTEXT, RIGHT_CONTEXT, self.N_FEATURES)

                    # estimate the action to be taken for i, i+ 1 target  nodes
                    y = self.estimate_train_action(trees, i)
                    
                    self.actions[y] += 1
                    self.update_pos_action(tree_pos_tag, y)

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

        train_tags = train_x.keys()

        # mainly for testing
        # if dummy is true we look at only 2 tags
        # if(dummy):
        #     train_tags = ['PRP$','VBG']

        # train classifier for each train tag
        for lp in train_tags:
            print lp
            print len(train_x[lp])
            print self.N_FEATURES

            # set of unique classes for each tag
            n_classes = Set()
            for i in train_y[lp]:
                n_classes.add(i)
            
            # train only if there are at least 2 classes
            # else the classifier would fail
            if( len(n_classes) > 1 ):

                clf_file = self.model_path + lp+".p"
                # load if the classifier already exists on disk
                # else train
                if os.path.isfile(clf_file):
                    print "load: "+ clf_file
                    clf[lp] = pickle.load( open( clf_file, "rb" ) )
                    train_x[lp] = None
                else:
                    temp_features = lil_matrix((len(train_x[lp]), self.N_FEATURES), dtype = bool)
                    
                    for i in range( 0, len(train_x[lp]) ):
                        for k in train_x[lp][i]:
                            temp_features[ i,k ] = True

                    features[lp] = temp_features.tocsr()

                    train_x[lp] = None

                    # cross validation code, really slow
                    # tuned_parameters = [{'kernel': ['poly'], 'C': [1, 10, 100]}]
                    # clf[lp] = GridSearchCV(svm.SVC(kernel='poly', degree=1, gamma=1, coef0=1, cache_size=5120), tuned_parameters, cv=3, n_jobs=3)
                    
                    clf[lp] = svm.SVC(kernel='poly', degree=2, gamma=1, coef0=1, cache_size=self.cache_size)
                    # clf[lp] = svm.LinearSVC()
                    
                    clf[lp].fit(features[lp], train_y[lp])
                    # print(clf[lp].best_params_)
                    
                    print "pickle: "+ clf_file
                    pickle.dump(  clf[lp] , open( clf_file, "wb" ) )


        # write the full model to disk
        self.clf = clf
        print "pickling"
        pickle.dump( clf , open( self.model_path + FINAL_MODEL+".p", "wb" ) )
        print "pickling done"

    def test(self, sentences):
        """
        Inference on test sentences
        input: test sentences
        """
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
                    extracted_features = self.extract_test_features(trees, i, LEFT_CONTEXT, RIGHT_CONTEXT, self.N_FEATURES)

                    # estimate the action to be taken for i, i+ 1 target  nodes
                    y = self.estimate_action(trees, i, extracted_features)
                    i, trees = self.take_action(trees, i ,y)
                    # execute the action and modify the trees
                    if y!= SHIFT:
                        no_construction = False
                    self.test_actions[y] += 1
            if(len(trees) == 1):
                total+=1
                # print total
            inferred_trees += [trees]

        return inferred_trees


    def dummy_test(self, sentences):
        """
        to test with limited set of tags
        not used in the main workflow
        """
        print len(sentences)
        test_sentences = []
        true_sentences = []
        inferred_trees = []
        accuracy_n = {'PRP$':0, 'VBG':0}
        accuracy_d = {'PRP$':0, 'VBG':0}
        total = 0
        for s in sentences:
            true_sentences += [sentence.ParsedSentence( s.words, s.pos_tags, s.dependency )]

        for s in true_sentences:
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
                    extracted_features = self.extract_test_features(trees, i, LEFT_CONTEXT, RIGHT_CONTEXT, self.N_FEATURES)

                    # estimate the action to be taken for i, i+ 1 target  nodes
                    y = self.estimate_train_action(trees, i)
                    tree_pos_tag = self.get_pos( trees, i )
                    if tree_pos_tag in ['PRP$','VBG']:
                        accuracy_d[tree_pos_tag] += 1
                        y_prime = self.estimate_action(trees, i, extracted_features)
                        if y == y_prime:
                            accuracy_n[tree_pos_tag] += 1


                    i, trees = self.take_action(trees, i ,y)
                    # execute the action and modify the trees
                    if y!= SHIFT:
                        no_construction = False
                    self.test_actions[y] += 1
            if(len(trees) == 1):
                total+=1
                # print total
            inferred_trees += [trees]
            # print len(inferred_trees)

        print counter_ratio(accuracy_n,accuracy_d)
        print accuracy_n, accuracy_d

        print self.test_actions
        print total
        return inferred_trees


    def evaluate(self, inferred_trees, gold_sentences):
        """
        Method to evaluate the parsing. 3 metrics
        Dependency accuracy (DA): The proportion
        of non-root words that are assigned the correct
        head

        Root accuracy (RA): The proportion of root
        words that are analyzed as such 

        Complete match (CM): The proportion of
        sentences whose unlabeled dependency structure
        is completely correct

        """
        PUNCTUATION_TAGS = [',','.',':','\'\'','``','PUNCT','SYM']
        root_accuracy_n = Counter()
        # root_accuracy_d = Counter()

        dep_accuracy_n = Counter()
        dep_accuracy_d = Counter()

        complete_d = 0
        complete_n = 0
        
        total_sentences = len(gold_sentences)
        for i,it in enumerate(inferred_trees):
            s = gold_sentences[i]

            # if there is only one tree
            if(len(it) == 1):
                complete_d += 1
                # get the current root
                current_root = it[0]
                # increment root accuracy denominator
                # root_accuracy_d[current_root.lex] += 1

                # check if the entire tree matches
                if it[0].match_all(s):
                    # update number of complete correct parsed trees
                    complete_n += 1

                # if current root is not a punctuation
                if current_root.pos_tag not in PUNCTUATION_TAGS:
                    # check if it is the tree root is same as the gold sentence root
                    if( s.dependency[current_root.position] == -1 ):
                        root_accuracy_n[current_root.lex] += 1

            # multiple trees
            else:
                for t in it:
                    current_root = t
                    # root_accuracy_d[current_root.lex] += 1
                    if current_root.pos_tag not in PUNCTUATION_TAGS:
                        if( s.dependency[current_root.position] == -1 ):
                            root_accuracy_n[current_root.lex] += 1

            # count how many nodes have correct parents ignoring punctuation
            for t in it:
                t.match_dep(s,dep_accuracy_n,dep_accuracy_d)


        print "root accuracy: " + str(np.sum(root_accuracy_n.values())/total_sentences)
        print "dep accuracy: " + str(np.sum(dep_accuracy_n.values())/np.sum(dep_accuracy_d.values()))
        print "complete: " + str(complete_n/complete_d)