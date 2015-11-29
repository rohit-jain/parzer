import StanfordDependencies
sd = StanfordDependencies.get_instance(backend='subprocess')

FILE_PATH = "/Users/rohitjain/github/nlp/dp/data/genia-dist/division/"

def readfile(filepath): 
    tokens = []
    trees = []
    n = 0
    OUTPUT_FILE = open(FILE_PATH + "test", "w")

    with open(filepath, 'r') as f: 
        for line in f:
            for i in sd.convert_tree(line):
                OUTPUT_FILE.write("\t".join([i.form, i.cpos, str(i.head - 1)]) + "\n")
            OUTPUT_FILE.write("\n")
            n += 1
            if (n%10)==0:
                print n
    OUTPUT_FILE.close()
    print "converted"

readfile(FILE_PATH + "test.trees")