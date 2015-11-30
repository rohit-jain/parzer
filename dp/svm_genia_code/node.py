class Node(object):
    """
    Node of a tree
    Can contain any number of children
    left and right represents the children created due to left and right dependencies respectively
    """
    def __init__(self, lex = None, position = None, pos_tag = None, dependency = -1, left = [], right = []):
        self.lex = lex
        self.position = position
        self.pos_tag = pos_tag
        self.dependency = dependency
        self.left = left
        self.right = right

    def insert_right(self, child):
        child.dependency = self.position
        self.right = self.right + [child]

    def insert_left(self, child):
        child.dependency = self.position
        self.left = self.left + [child]

    def match_all(self, gold_sentence):
        correct_roots = self.match(gold_sentence)
        if( correct_roots == len(gold_sentence.words)):
            return True
        else:
            return False

    def match(self, gold_sentence):
        PUNCTUATION_TAGS = [',','.',':','\'\'','``']
        correct_roots = 0
        position = self.position
        dep = self.dependency
        tag = gold_sentence.pos_tags[position]
        if(( gold_sentence.dependency[position] == dep ) or (tag in PUNCTUATION_TAGS)):
            correct_roots += 1

        if len(self.right) > 0:
            for r in self.right:
                correct_roots += r.match(gold_sentence)

        if len(self.left) > 0:
            for l in self.left:
                correct_roots += l.match(gold_sentence)

        return correct_roots

    def match_dep(self, gold_sentence, result_dict, baseline_dict):
        PUNCTUATION_TAGS = [',','.',':','\'\'','``']
        correct_roots = 0
        position = self.position
        dep = self.dependency
        word = gold_sentence.words[position]
        tag = gold_sentence.pos_tags[position]

        if (dep!=-1) and (tag not in PUNCTUATION_TAGS):
            baseline_dict[word] += 1
            if( gold_sentence.dependency[position] == dep ):
                correct_roots += 1
                result_dict[gold_sentence.words[position]] += 1

        if len(self.right) > 0:
            for r in self.right:
                correct_roots += r.match_dep(gold_sentence, result_dict, baseline_dict)


        if len(self.left) > 0:
            for l in self.left:
                correct_roots += l.match_dep(gold_sentence, result_dict, baseline_dict)

        return correct_roots

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
        return "lex: %s, pos_tag: %s, dependency: %s, position: %s, left: %s, right: %s\n" % (self.lex, self.pos_tag, self.dependency, self.position , "[" + "\n".join(temp_left) + "]", "[" + "\n".join(temp_right) + "]")
    
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