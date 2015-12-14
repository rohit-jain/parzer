import logging
import os
import sentence
import dependency_parser
import collections
import time
import getopt
import sys
from functools import wraps
from sets import Set

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


def read_train( path, filename ):
    """
    Read training sentences from files 
    converted as per original format
    """
    sentences = []

    words = []
    pos_tags = []
    dependencies = []
    for line in open( path + filename , "r"):
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

    return sentences


def read_penn_treebank( path, low, high ):
    """
    Read training sentences from dependency 
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


def read_test( filename ):
    """
    Read sentences from files tagged and 
    converted as per original format
    """
    sentences = []

    words = []
    pos_tags = []
    dependencies = []
    tags = []
    for line in open( filename, "r"):
        if (line != "\n"):
            word, tag, pos_tag, dependency = line.strip("\n").split("\t")
            words += [ word ]
            tags += [ tag ]
            pos_tags += [ pos_tag ]
            dependencies += [ int(dependency) ]
        else:
            sentences += [sentence.ParsedSentence( words, tags, dependencies )]
            words = []
            tags = []
            pos_tags = []
            dependencies = []

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
                        sentences += [sentence.ParsedSentence( words, tags, dependencies )]
                        words = []
                        tags = []
                        pos_tags = []
                        dependencies = []
        s += 1
    return sentences


@profile
def main():
    DIR_PATH = os.getcwd()
    # directory paths for penn treebank dataset
    DATA_PATH = DIR_PATH + "/data/wsj_parsed/"
    ST_DATA_PATH = DIR_PATH + "/data/st_tagged/"
    GT_DATA_PATH = DIR_PATH + "/data/gt_tagged/"

    # directory paths for GENIA corpus
    GEN_DATA_PATH = DIR_PATH + "/data/genia-dist/division/"
    # stanford tagged genia test file
    GEN_STAN_TAG = "tagged_stan_test"
    # genia tagger tagged genia test file
    GEN_GDEP_TAG = "tagged_test"

    # directory path for spanish
    ES_DATA_PATH = DIR_PATH + "/data/spanish/"
    # stanford tagged spanish test file
    ES_STAN_TAG = "test_correct_tagged"

    MODEL_DIR = DIR_PATH + "/models/"
    cache_size = 5120
    dataset = ""
    tagger = ""

    myopts, args = getopt.getopt(sys.argv[1:], "c:i:t:")
    for o, a in myopts:
        if o == '-c':
            cache_size = int(a)
        elif o == '-i':
            dataset = a
        elif o == '-t':
            tagger = a
    
    # dataset based on the command line input
    if(dataset == "ptb"):
        train_data_path = DATA_PATH
    elif(dataset == "genia"):
        train_data_path = GEN_DATA_PATH
    elif(dataset == "spanish"):
        train_data_path = ES_DATA_PATH

    # tagger based on the command line input
    if(tagger == "stanford"):
        if(train_data_path == DATA_PATH):
            test_data_path = ST_DATA_PATH
        elif(train_data_path == GEN_DATA_PATH):
            test_data_path = GEN_DATA_PATH + GEN_STAN_TAG
        elif(train_data_path == ES_DATA_PATH):
            test_data_path = ES_DATA_PATH + ES_STAN_TAG

    elif(tagger == "gdep"):
        if(train_data_path == DATA_PATH):
            test_data_path = GT_DATA_PATH
        elif(train_data_path == GEN_DATA_PATH):
            test_data_path = GEN_DATA_PATH + GEN_GDEP_TAG
        elif(train_data_path == ES_DATA_PATH):
            print "GDEP tagger cannot be used with spanish"
            sys.exit(0)

    # model path based on the command line input
    MODEL_PATH = MODEL_DIR + dataset + "_" + tagger + "/"

    # Read train sentences from penn treebank for the given sections with labels
    # logging.info("Reading training data")
    if(dataset == "ptb"):
        training_sentences = read_penn_treebank(train_data_path, "0200", "2199")
    else:
        training_sentences = read_train(train_data_path,"train")

    # Read validate sentences from penn treebank for the given sections without labels
    if(dataset == "ptb"):
        test_sentences = read_test_penn_treebank(test_data_path, "2300", "2399")
    else:
        test_sentences = read_test(test_data_path)

    # Initialise parser
    my_parser = dependency_parser.SVMParser(model=MODEL_PATH, load=True, cache_size=cache_size)
    
    # train the data
    my_parser.train( training_sentences )

    # inference
    inferred_trees = my_parser.test ( test_sentences )

    # evaluation
    my_parser.evaluate( inferred_trees, test_sentences )

if __name__ == '__main__':
    main()
    print_prof_data()