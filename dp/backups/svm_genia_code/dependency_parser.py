from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn import svm
# from nltk.tag import StanfordPOSTagger
import sentence
import pickle, random
from sets import Set
from collections import Counter
import os
from sklearn.grid_search import GridSearchCV

LEFT = 0
SHIFT = 1
RIGHT = 2
LEFT_CONTEXT = 2
RIGHT_CONTEXT = 4
GUESS = Counter()

def counter_ratio(n,d):
    r = dict()
    for i in n:
        r[i] = n[i]/d[i]
    return r

def convert_to_features(d):
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
        print pi,l
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
    def __init__(self, load=False):
        Parser.__init__(self)
        self.st = {}#StanfordPOSTagger("wsj-0-18-bidirectional-distsim.tagger")
        self.clf = {}
        self.position_vocab = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.position_tag = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.ch_l_tag = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.ch_r_tag = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.ch_l_vocab = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}
        self.ch_r_vocab = {0:Counter(), 1:Counter(), 2:Counter(), 3:Counter(), 4:Counter(), 5:Counter(), 6:Counter(), 7:Counter()}

        self.loaded = False
        self.actions = Counter()
        self.test_actions = Counter()
        self.target_feature_size = None #(3 * len(self.vocab)) + (3 * len(self.tags))
        self.context_feature_size = None #( len(self.vocab) + len(self.tags) )
        self.N_FEATURES = None #(LEFT_CONTEXT + RIGHT_CONTEXT) * self.context_feature_size + 2 * self.target_feature_size
        if load == True:
            self.loaded = True
            self.clf = pickle.load( open( "full_svm.p", "rb" ) )
            self.N_FEATURES = 4242238
        

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
            try:
                action_array = self.clf[tree_pos_tag].predict( temp_features )
            except Exception as e:
                print tree_pos_tag
                print e
                print "guess"
                action_array = [SHIFT, LEFT, RIGHT]
                return action_array[0]
        else:
            GUESS[tree_pos_tag] += 1
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

    def lex_feature( self, position, node, offset ):
        vocab = self.position_vocab[position]
        lex_index = vocab[("<UNKNOWN>")]
        if (node!=[]):
            if ((node.lex) in vocab):
                lex_index = vocab[(node.lex)]
        return lex_index + offset

    def pos_feature( self, position, node, offset ):
        tag = self.position_tag[position]
        tag_index = tag[("<UNKNOWN>")]
        if (node!=[]):
            if ( node.pos_tag in tag):
                tag_index = tag[(node.pos_tag)]
        return tag_index + offset

    def child_feature( self, position, node, offset, family ):
        vocab = family
        lex_index = vocab[("<UNKNOWN>")]
        if (node!=[]):
            if ((node.lex) in vocab):
                lex_index = vocab[(node.lex)]
        return lex_index + offset

    def child_lex( self, position, children, offset, family ):
        lex_indices = []
        for child in children:
            lex_indices += [self.child_feature( position,child,offset,family )]
        return lex_indices

    # def child_pos( self, children, offset ):
    #     pos_indices = []
    #     for child in children:
    #         pos_indices += [self.pos_feature( child,offset )]
    #     return pos_indices


    # def node_features( self, target_node, offset, child_features=False ):
    #     v = len(self.vocab)
    #     t = len(self.tags)
    #     lex = [self.lex_feature( target_node,0 ) + offset]
    #     pos = [self.pos_feature( target_node,v ) + offset]
    #     if child_features == True:
    #         ch_l_lex = [i+offset for i in self.child_lex( target_node.left,v+t ) ]
    #         ch_l_pos = [i+offset for i in self.child_pos( target_node.left,2*v+t ) ]
    #         ch_r_lex = [i+offset for i in self.child_lex( target_node.right,(2*v)+(2*t) ) ]
    #         ch_r_pos = [i+offset for i in self.child_pos( target_node.right,(3*v)+(2*t) ) ]
    #         return lex + pos + ch_l_lex + ch_l_pos + ch_r_lex + ch_r_pos
    #     return lex + pos

    # def extract_mat_features(self, feature_nodes):
    #     features = []
    #     print len(feature_nodes)
    #     for fn in feature_nodes:
    #         # print fn.keys()
    #         temp_features = []
    #         offset = 0
    #         for position in fn:
    #             temp_lex = self.lex_feature(position, fn[position], offset)
    #             offset += len(self.position_vocab[position])
    #             temp_tag = self.pos_feature(position, fn[position], offset)
    #             offset += len(self.position_tag[position])
    #             temp_features += [ temp_lex, temp_tag ]
    #         features += [temp_features]
    #     return features

    def extract_test_features(self, trees, i, l, r):
        features = []
        offset = 0
        # features_tag = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
        for k,w in enumerate(range(i-l,(i+1+r+1))):
            if( w>= 0) and ( w< len(trees)):
                target_node = trees[w]
                temp_lex = [self.lex_feature(k, target_node, offset)]
                offset += len(self.position_vocab[k])
                temp_tag = [self.pos_feature(k, target_node, offset)]
                offset += len(self.position_tag[k])
                temp_ch_l_lex = self.child_lex(k, target_node.left, offset, self.ch_l_vocab[k])
                offset += len(self.ch_l_vocab[k])
                temp_ch_l_tag = self.child_lex(k, target_node.left, offset, self.ch_l_tag[k])
                offset += len(self.ch_l_tag[k])
                temp_ch_r_lex = self.child_lex(k, target_node.right, offset, self.ch_r_vocab[k])
                offset += len(self.ch_r_vocab[k])
                temp_ch_r_tag = self.child_lex(k, target_node.right, offset, self.ch_r_tag[k])
                offset += len(self.ch_r_tag[k])

                features += (temp_lex + temp_tag + temp_ch_r_tag + temp_ch_r_lex + temp_ch_l_tag + temp_ch_l_lex )
        return features


    def build_vocab(self, trees, i, l, r):
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

                # if ( k<l ):
                #     features += self.node_features( target_node,k*self.context_feature_size )
                # elif ( k>=l and k<l+2 ):
                #     target_offset = 2*self.context_feature_size + (k-l)*self.target_feature_size
                #     features += self.node_features( target_node,target_offset,child_features=True )
                # else:
                #     target_offset = (2 + k - (l+2) )*self.context_feature_size + 2*self.target_feature_size
                #     features += self.node_features( target_node,target_offset )

    def get_pos(self, trees, i):
        target_node = trees[i]
        return target_node.pos_tag

    def train(self, sentences, sentences2):
        m = len(sentences)
        print "Train Sentences: " + str(m) + "," + str(len(sentences2))
        train_x = {}
        train_y = {}
        features = {}
        clf = {}

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

        # convert counter to features
        self.position_tag = convert_to_features(self.position_tag)
        self.position_vocab = convert_to_features(self.position_vocab)
        self.ch_l_tag = convert_to_features(self.ch_l_tag)
        self.ch_r_tag = convert_to_features(self.ch_r_tag)
        self.ch_l_vocab = convert_to_features(self.ch_l_vocab)
        self.ch_r_vocab = convert_to_features(self.ch_r_vocab)

        self.N_FEATURES = count_features(self.position_tag) + count_features(self.position_vocab) + count_features(self.ch_l_tag) + count_features(self.ch_l_vocab) + count_features(self.ch_r_tag) + count_features(self.ch_r_vocab)
        print self.N_FEATURES

        # convert dummy to real
        # for pos_tag in dummy_train_x:
        #     train_x[pos_tag] = self.extract_mat_features(dummy_train_x[pos_tag])
        if(self.loaded):
            return

        for s in sentences2:
            trees = s.get_labeled_trees()
            # print "here"
            # print len(trees)
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
                    extracted_features = self.extract_test_features(trees, i, LEFT_CONTEXT, RIGHT_CONTEXT)

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


        print self.actions
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
                clf_file = lp+".p"
                if os.path.isfile(clf_file):
                    print "load: "+ clf_file
                    clf[lp] = pickle.load( open( clf_file, "rb" ) )
                else:
                    #tuned_parameters = [{'kernel': ['poly'], 'C': [1, 10, 100]}]
                    #clf[lp] = GridSearchCV(svm.SVC(kernel='poly', degree=1, gamma=1, coef0=1, cache_size=5120), tuned_parameters, cv=3, n_jobs=3)
                    clf[lp] = svm.SVC(kernel='poly', degree=2, gamma=1, coef0=1, cache_size=7168)
                    # clf[lp] = svm.LinearSVC()
                    clf[lp].fit(features[lp], train_y[lp])
                    # print(clf[lp].best_params_)
                    print "pickle: "+ clf_file
                    pickle.dump( clf[lp] , open( lp+".p", "wb" ) )


        self.clf = clf
        print "pickling"
        pickle.dump( clf , open( "full_svm.p", "wb" ) )
        print "pickling done"

    def test(self, sentences):
        print len(sentences)
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
                    extracted_features = self.extract_test_features(trees, i, LEFT_CONTEXT, RIGHT_CONTEXT)

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
            # print len(inferred_trees)

        print self.test_actions
        print total
        print GUESS
	return inferred_trees


    def evaluate(self, inferred_trees, gold_sentences):
        # Dependency accuracy (DA): The proportion
        # of non-root words that are assigned the correct
        # head

        # Root accuracy (RA): The proportion of root
        # words that are analyzed as such 

        # Complete match (CM): The proportion of
        # sentences whose unlabeled dependency structure
        # is completely correct
        PUNCTUATION_TAGS = [',','.',':','\'\'','``']
        root_accuracy_n = Counter()
        root_accuracy_d = Counter()
        root_accuracy = dict()

        dep_accuracy_n = Counter()
        dep_accuracy_d = Counter()
        dep_accuracy = dict()

        complete_d = 0
        complete_n = 0
        
        total_sentences = len(gold_sentences)
        for i,it in enumerate(inferred_trees):
            s = gold_sentences[i]

            if(len(it) == 1):
                complete_d += 1
                current_root = it[0]
                root_accuracy_d[current_root.lex] += 1

                if it[0].match_all(s):
                    complete_n += 1

                if current_root.pos_tag not in PUNCTUATION_TAGS:
                    if( s.dependency[current_root.position] == -1 ):
                        root_accuracy_n[current_root.lex] += 1
           
            else:
                for t in it:
                    current_root = t
                    root_accuracy_d[current_root.lex] += 1
                    if current_root.pos_tag not in PUNCTUATION_TAGS:
                        if( s.dependency[current_root.position] == -1 ):
                            root_accuracy_n[current_root.lex] += 1


            for t in it:
                t.match_dep(s,dep_accuracy_n,dep_accuracy_d)


        # 3rd w else, 1st without else
        root_accuracy = counter_ratio(root_accuracy_n,root_accuracy_d)
        print "root accuracy: " + str(np.sum(root_accuracy_n.values())/np.sum(root_accuracy_d.values())) + "," + str(np.mean(root_accuracy.values())) + "," + str(np.sum(root_accuracy_n.values())/total_sentences)
        dep_accuracy = counter_ratio(dep_accuracy_n,dep_accuracy_d)
        print "dep accuracy: " + str(np.sum(dep_accuracy_n.values())/np.sum(dep_accuracy_d.values())) + "," + str(np.mean(dep_accuracy.values()))
        print "complete: " + str(complete_n/complete_d)
