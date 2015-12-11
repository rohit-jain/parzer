import logging
import os
import sentence
import dependency_parser
import collections
import time
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


def read_test( path, filename ):
    """
    Read sentences from files tagged and 
    converted as per original format
    """
    sentences = []

    words = []
    pos_tags = []
    dependencies = []
    tags = []
    for line in open( path + filename, "r"):
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
    DATA_PATH = DIR_PATH + "/data/wsj_parsed/"
    ST_DATA_PATH = DIR_PATH + "/data/st_tagged/"
    GEN_DATA_PATH = DIR_PATH + "/data/genia-dist/division/"
    ES_DATA_PATH = DIR_PATH + "/data/spanish/"
    GT_DATA_PATH = DIR_PATH + "/data/gt_tagged/"
    
    # Read train sentences from penn treebank for the given sections with labels
    # logging.info("Reading training data")
    # training_sentences = read_penn_treebank(DATA_PATH, "0200", "2199")
    training_sentences = read_train(ES_DATA_PATH,"train")

    # Read validate sentences from penn treebank for the given sections without labels
    # validation_sentences = read_test_penn_treebank(ST_DATA_PATH, "2300", "2399")
    validation_sentences = read_test(ES_DATA_PATH, "test_tagged")

    # Initialise parser
    my_parser = dependency_parser.SVMParser(load=False, cache_size=int(sys.argv[1]))
    
    # train the data
    # logging.info("train")
    # my_parser.train( training_sentences, dummy=False )
    my_parser.train( training_sentences )
    # my_parser.tag( validation_sentences )
    

    # print "infer"
    inferred_trees = my_parser.test ( validation_sentences )
    my_parser.evaluate( inferred_trees, validation_sentences )

if __name__ == '__main__':
    main()
    print_prof_data()