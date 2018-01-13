import math
import metapy
import numpy as np
from numpy.linalg import norm
from svd_embeddings import SVDEmbeddings
from scipy import stats
import sys

def compute_reciprocal_rank(word, candidates):
    ##### IMPLEMENT ME! #####
    try:
        r = candidates.index(word)
        return 1.0/(r+1)
    except:
        return 0.0


def top_k(emb, w_x, w_y, w_z, k=10):
    x = glove.at(w_x)
    y = glove.at(w_y)
    z = glove.at(w_z)

    if emb != glove:
        query = emb.at(y[0]) - emb.at(x[0]) + emb.at(z[0])
    else:
        query = y[1] - x[1] + z[1]

    query /= norm(query, 2)

    # query for k + 3 to accomodate for the query words possibly being in
    # the top k words
    results = emb.top_k(query, k + 3)
    # filter out all query words
    results = [result for result in results if not result[0] in (x[0],
        y[0], z[0])]
    # limit to top 10
    results = results[0:10]

    return [glove.term(result[0]) for result in results]


def compute_rr(emb):
    rr = []
    with open("analogies.txt") as f:
        for line in f:
            line = line.strip()
            words = line.split()

            candidates = top_k(emb, words[0], words[1], words[2])
            rr.append(compute_reciprocal_rank(words[3], candidates))
    print("Number of analogies tested: {}".format(len(rr)))
    print("MRR: {} (+/- {})".format(np.mean(rr), np.std(rr)))
    return rr


def load_embedding(name):
    if name == 'glove':
        return glove
    else:
        return SVDEmbeddings(name)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} embeddings1 [embeddings2]".format(sys.argv[0]))
        print("\tembeddings(1|2) are either .npy files or \"glove\"")
        sys.exit(1)

    # GloVe is always loaded in order to use its vocabulary mapping
    glove = metapy.embeddings.load_embeddings('config.toml')

    recip_ranks = []
    for method in sys.argv[1:]:
        emb = load_embedding(method)
        print("Testing {}...".format(method))
        recip_ranks.append(compute_rr(emb))

    if len(recip_ranks) == 2:
        res=stats.ttest_rel(recip_ranks[0], recip_ranks[1]) #Two tailed t test
        print ('Two tailed test t value = ',res[0] )
        print ('\n p value = ',res[1])
        
        ##### FILL IN THE CODE HERE #####
        #
        # Determine whether the performance difference between the two
        # methods is statistically significant by using a t-test.
        #

        #
        ##### STOP FILLING IN THE CODE HERE #####
