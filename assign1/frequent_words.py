import sys

def read_data(word_collection):

    """ 
    Read data from the files for the given collection and compute a map of doc_id to array of words 
    Returns words, doc_vector map
    """

    words = []
    word_vector = {}

    with open("./data/vocab."+word_collection+".txt") as vocabfile:
        for line in vocabfile.readlines():
            words.append(line.strip())

    with open("./data/docword."+word_collection+".txt") as docwordfile:
        (num_docs, num_words, nnz) = (docwordfile.readline(), docwordfile.readline(), docwordfile.readline())
        num_docs = int(num_docs.strip())
        num_words = int(num_words.strip())
        nnz = int(nnz.strip())

        for i in range(num_words):
            word_vector[i] = set()

        for i in range(nnz):
            line = docwordfile.readline()
            line = line.strip()
            [doc_id, word_id, count] = line.split(' ')
            doc_id = int(doc_id) - 1
            word_id = int(word_id) - 1
            
            word_vector[word_id].add(doc_id)
            
    return (words, word_vector)

def validate_args(args):
    """
    Validates the command line arguments and parses them
    """
    if len(args) != 4:
        print("Usage: python frequent_words.py <k> <f> <collection>")
        sys.exit(1)
    try:
        k = int(args[1])
        f = int(args[2])
        word_collection = args[3]
    except Exception:
        print("Invalid input, unable to parse.")
        print("Usage: python frequent_words.py <k> <f> <collection>")
        sys.exit(1)

    valid_collections = ["nips", "kos", "enron"]
    if word_collection not in valid_collections:
        print("Invalid collection. Must be one of " + ','.join(valid_collections))
    
    return (k, f, word_collection)

class Word:
    def __init__(self, word, id, docs):
        self.w = word
        self.id = id
        self.docs = docs

    def __eq__(self, value):
        return self.w == value.w and self.id == value.id
    
    def __ne__(self, value):
        return self.w != value.w or self.id != value.id

    def __str__(self):
        return self.w

class WordSet:
    def __init__(self, words):
        self.words = list(words)
        self.k = len(words)
        self.count = 0
    
    def add_word(self, word):
        self.words.append(word)
        self.k = self.k + 1

    def get_key(self):
        return tuple(word.id for word in self.words)

    def compute_count(self):
        i = self.words[0].docs
        for word in self.words:
            i = i.intersection(word.docs)
        self.count = len(i)

    def set_count(self, count):
        self.count = count

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise Exception("Invalid access")
        
        if index >= self.k or index < 0:
            raise Exception("Index out of bounds")
    
        return self.words[index]
    
    def get_words(self):
        return self.words + []

    def __str__(self):
        words = []
        for w in self.words:
            words.append(w.w)
        return "Count = " + str(self.count) + ", Words = " + ",".join(words)

def frequent_sets(words, word_doc_map, k , f):
    
    def get_count(word_set):
        count = 0
        for doc, doc_words in word_doc_map.items():
            for word in word_set.get_words():
                if word.id not in doc_words:
                    break
            else:
                count = count + 1
        return count
        
    def differ_by_one(word_set1, word_set2):
        k = word_set1.k
        for i in range(k-1):
            if word_set1[i] != word_set2[i]:
                return False
        return word_set1[k-1] != word_set2[k-1]

    def gen_candidates(cur_freq_set, k):
        candidates = []
        cur_freq_set.sort(key = lambda x: x.get_key())
        l = len(cur_freq_set)
        for i in range(l):
            j = i + 1
            while j < l and differ_by_one(cur_freq_set[i], cur_freq_set[j]):
                temp = WordSet(cur_freq_set[i].get_words())
                temp.add_word(cur_freq_set[j][k-1])
                candidates.append(temp)
                j = j + 1
        return candidates

    candidates = [WordSet([Word(word, index, word_doc_map[index])]) for index, word in enumerate(words)]
    for ws in candidates:
        ws.compute_count()
    frequent_set = list(filter(lambda x: x.count >= f, candidates))
    i = 1

    while i<k and frequent_set:
        print("Generating "+str(i+1)+" candidate sets ...")
        candidates = gen_candidates(frequent_set, i)
        for ws in candidates:
            ws.compute_count()
        frequent_set = list(filter(lambda x: x.count >= f, candidates))
        i = i + 1
    
    return frequent_set

def main():
    
    (k, f, word_collection) = validate_args(sys.argv)
    (words, word_vector) = read_data(word_collection)

    print("Generating frequent sets ...")
    f_sets = frequent_sets(words, word_vector, k, f)
    if len(f_sets) != 0:
        print("Size of frequent sets : ", len(f_sets))
        print("The frequent sets are : ")
        for ws in f_sets:
            print(ws)
    else:
        print("No frequent sets satisfy the frequency threshold")

if __name__ == "__main__":
    main()