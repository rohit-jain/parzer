import logging
import os
import sentence
import dependency_parser
import collections
import time
from nltk.tag import StanfordPOSTagger
from functools import wraps

PROF_DATA = {}
LOG_FILENAME = 'logging.out'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG,)

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling

def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print "Function %s called %d times. " % (fname, data[0]),
        print 'Execution time max: %.3f, average: %.3f' % (max_time, avg_time)

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}


def extract_vocabulary_tags(sentences):
    vocab = collections.Counter()
    tags = collections.Counter()
    for s in sentences:
        vocab.update(s.words)
        tags.update(s.pos_tags)
    v = {}
    i = 0
    for k in vocab.keys():
        v[k] = i
        i += 1
    t = {}
    i = 0
    for k in tags.keys():
        t[k] = i
        i += 1
    v["<UNKNOWN>"] = i
    t["<UNKNOWN>"] = i
    return v,t


def read_penn_treebank( path, low, high ):
    """
    Read sentences from dependency 
    converted penn treebank files
    """
    s = int(low[0:2])
    e = int(high[0:2])
    sentences = []

    while( s <= e ):
        segment = path + str(s).zfill(2)
        for f in os.listdir(segment):
            file_path = segment + "/" + f
            if (not f.startswith('.')) and os.path.isfile( file_path ) and (int(f[-4:]) <= int(high)):
                words = []
                pos_tags = []
                dependencies = []
                for line in open( file_path , "r"):
                    if (line != "\n"):
                        word, tag, dependency = line.strip("\n").split("\t")
                        words += [ word ]
                        pos_tags += [ tag ]
                        dependencies += [ int(dependency) ]
                    else:
                        sentences += [sentence.ParsedSentence( words, pos_tags, dependencies )]
                        words = []
                        pos_tags = []
                        dependencies = []
        s += 1
    return sentences

def read_test_penn_treebank( path, low, high ):
    """
    Read sentences from dependency 
    converted and stanford tagged penn treebank files
    """
    s = int(low[0:2])
    e = int(high[0:2])
    sentences = []

    while( s <= e ):
        segment = path + str(s).zfill(2)
        for f in os.listdir(segment):
            file_path = segment + "/" + f
            if (not f.startswith('.')) and os.path.isfile( file_path ) and (int(f[-4:]) <= int(high)):
                words = []
                pos_tags = []
                dependencies = []
                tags = []
                for line in open( file_path , "r"):
                    if (line != "\n"):
                        word, tag, pos_tag, dependency = line.strip("\n").split("\t")
                        words += [ word ]
                        tags += [ tag ]
                        pos_tags += [ pos_tag ]
                        dependencies += [ int(dependency) ]
                    else:
                        # test with only one sentence
                        # if len(sentences) == 1:
                        #   break
                        # if len(words) >= 6 and len(words) <= 14:
                        sentences += [sentence.ParsedSentence( words, tags, dependencies )]
                        words = []
                        tags = []
                        pos_tags = []
                        dependencies = []
        s += 1
    return sentences


def tag_penn_treebank( path, low, high, path2 ):
    """
    stan pos penn treebank files
    """
    st = StanfordPOSTagger("wsj-0-18-bidirectional-distsim.tagger")
    s = int(low[0:2])
    e = int(high[0:2])
    sentences = []

    while( s <= e ):
        segment = path + str(s).zfill(2)
        store_segment = path2 + str(s).zfill(2)
        for f in os.listdir(segment):
            if not os.path.exists( store_segment ):
                os.makedirs( store_segment )

            # file path for reading and writing
            file_path = segment + "/" + f
            store_file_path = store_segment + "/" + f

            if (not f.startswith('.')) and os.path.isfile( file_path ) and (int(f[-4:]) <= int(high)):
                words = []
                pos_tags = []
                dependencies = []
                tagged_file = open( store_file_path,"w" )
                for line in open( file_path ,"r"):
                    if (line != "\n"):
                        word, tag, dependency = line.strip("\n").split("\t")
                        words += [ word ]
                        pos_tags += [ tag ]
                        dependencies += [ int(dependency) ]
                    else:
                        word_tag_pairs = st.tag(words)
                        tags = [j for i,j in word_tag_pairs]
                        for i,w in enumerate(words):
                            tagged_file.write( "\t".join([w, tags[i], pos_tags[i], str(dependencies[i])]) + "\n" )
                        
                        tagged_file.write("\n")
                        sentences += [sentence.ParsedSentence( words, pos_tags, dependencies )]
                        words = []
                        pos_tags = []
                        dependencies = []
                tagged_file.close()
        s += 1
    return sentences


@profile
def main():
    DATA_PATH = "/Users/rohitjain/github/nlp/dp/data/wsj_parsed/"
    ST_DATA_PATH = "/Users/rohitjain/github/nlp/dp/data/st_tagged/"
    # Read train sentences from penn treebank for the given sections with labels
    logging.info("Reading training data")
    training_sentences = read_penn_treebank(DATA_PATH, "0200", "2199")
    # Read validate sentences from penn treebank for the given sections without labels
    valdation_sentences = read_test_penn_treebank(ST_DATA_PATH, "2300", "2399")

    training_vocabulary, training_tags = extract_vocabulary_tags(training_sentences)
    logging.info("Training Vocabulary: " + str(len(training_vocabulary)) + " Training Tags: " + str(len(training_tags)))
    
    # Initialise parser
    my_parser = dependency_parser.SVMParser(training_vocabulary, training_tags)
    # train the data
    logging.info("train")
    my_parser.train( training_sentences )
    # my_parser.tag( valdation_sentences )
    print "infer"
    inferred_trees = my_parser.test ( valdation_sentences )
    my_parser.evaluate( inferred_trees, valdation_sentences )

if __name__ == '__main__':
    main()
    print_prof_data()