from geniatagger import GeniaTagger

DIR_PATH = "/Users/rohitjain/github/nlp/dp/data/genia-dist/division/"

def tag_genia( path, path2, tagger_path ):
    """
    genia tagger
    """
    gt = GeniaTagger(tagger_path + 'geniatagger')
    sentences = []

    file_path = path
    store_file_path = path2
    n = 0
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
            n += 1
            if (n%10)==0:
                print n

            word_tag_pairs = gt.parse(" ".join(words))
            tags = [j[2] for j in word_tag_pairs]
            for i,w in enumerate(words):
                tagged_file.write( "\t".join([w, tags[i], pos_tags[i], str(dependencies[i])]) + "\n" )
            
            tagged_file.write("\n")
            words = []
            pos_tags = []
            dependencies = []
    tagged_file.close()
    print "done"


tag_genia(DIR_PATH+"dev",DIR_PATH+"tagged_dev","/Users/rohitjain/Downloads/geniatagger/")