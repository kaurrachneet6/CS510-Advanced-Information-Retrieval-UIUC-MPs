import metapy
import numpy as np
from scipy import stats
from svd_embeddings import SVDEmbeddings
import sys
from random import shuffle

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {} svd_file_p1 svd_file_p0.5 svd_file_p0".format(sys.argv[0]))
        sys.exit(1)

    svd_p10 = SVDEmbeddings(sys.argv[1])
    svd_p05 = SVDEmbeddings(sys.argv[2])
    svd_p00 = SVDEmbeddings(sys.argv[3])

    svds = [svd_p10, svd_p05, svd_p00]
    glove = metapy.embeddings.load_embeddings('config.toml')

    wins = [[], [], [], []]

    with open("similarity.txt") as f:
        for line in f:
            word = line.strip()
            print("Query: {}".format(word))

            word_id, glove_q = glove.at(word)
            svd_qs = [svd.at(word_id) for svd in svds]

            glove_c = [c[0] for c in glove.top_k(glove_q, 2) if c[0] != word_id]
            svd_cs = [[x[0] for x in lst if x[0] != word_id][0] for lst in
                     (svd.top_k(query, 2) for svd, query in zip(svds, svd_qs))]

            candidates = svd_cs + glove_c

            # want: map from idx -> methods returning that term
            terms = list(set(glove.term(tid) for tid in candidates))
            shuffle(terms)

            for i, term in enumerate(terms):
                print("{}) {}".format(i + 1, term))

            winner = None
            while winner == None or winner < 0 or winner > len(terms):
                winner = int(input('Winner: ')) - 1

            winning_id = glove.at(terms[winner])[0]
            for i, candidate in enumerate(candidates):
                wins[i].append(int(candidate == winning_id))

            print()

    print()
    print("Total questions: {}".format(len(wins[0])))
    print("SVD p=1.0: {}".format(np.mean(wins[0])))
    print("SVD p=0.5: {}".format(np.mean(wins[1])))
    print("SVD p=0.0: {}".format(np.mean(wins[2])))
    print("GloVe:     {}".format(np.mean(wins[3])))

    ##### FILL IN THE CODE HERE #####
    win_ratio= [np.mean(wins[0]), np.mean(wins[1]), np.mean(wins[2]), np.mean(wins[3])]
    ranking=np.argsort(win_ratio) #Sorting the indices acoording to win ratio
    res=stats.ttest_rel(wins[ranking[-1]], wins[ranking[-2]]) #Two tailed t test between best and second best method
    print ('Two tailed test t value = ',res[0] ) #T statistic
    print ('\n p value = ',res[1]) #p value
    
    #
    # Determine whether the best performing method is statistically
    # significantly different than the second best performing method by
    # using a t-test.

    ##### STOP FILLING IN THE CODE HERE #####
