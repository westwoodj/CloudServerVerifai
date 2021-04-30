
from utils import *


# create vocabulary

def myProcessing(tweet, vocab_file):
    #tweet = tweet.lower()
    d = re.split('\s', tweet[:-1])
    v = read_vocab(vocab_file)
    n = len(d)
    t = len(v)
    print(n, t)
    X = np.zeros((t,), dtype=np.double)
    for word in d:
        if word in v:
            print("word: ", word)
            X[v.index(word)] += 1
    #print(X)
    return X


if __name__ == '__main__':
    myProcessing("France: 10 people dead after shooting at HQ of satirical weekly newspaper #CharlieHebdo, according to witnesses http://t.co/FkYxGmuS58", 'vocab1.txt')