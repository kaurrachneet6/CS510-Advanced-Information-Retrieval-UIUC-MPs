import numpy as np
from numpy.linalg import norm
import scipy
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
import sys

# here are a few helper functions you might find useful...
def multiply_by_rows(matrix, coefficients):
    diag = diags(coefficients, 0)
    return diag * matrix

def multiply_by_columns(matrix, coefficients):
    diag = diags(coefficients, 0)
    return matrix * diag

if __name__ == '__main__':
    print('Loading cooccurrence matrix...', file=sys.stderr)
    with np.load('/data/cs510/mp2/cooccur.npz') as loader:
        PPMI = csr_matrix((loader['data'], loader['indices'],
            loader['indptr']), shape=loader['shape'])
    print('Computing PPMI...', file=sys.stderr)
    
    Mod_D=PPMI.data.sum()          # |D| - Sum of all entries of matrix
    n_c=PPMI.sum(axis=0).tolist()  # n_c - Sum of each of the columns
    n_w=PPMI.sum(axis=1)           # n_w - Sum of each of the rows
    n_w=n_w.T.tolist()
    nc_inv=np.reciprocal(list(map(float,n_c[0][:])))
    PPMI=multiply_by_columns(PPMI,nc_inv) #n_w,c / n_c
    nw_inv=np.reciprocal(list(map(float,n_w[0][:])))
    PPMI=multiply_by_rows(PPMI,nw_inv)    #n_w,c / n_c.n_w
    np.log2(PPMI.data*float(Mod_D), out=PPMI.data)
    
    ##### STOP FILLING IN THE CODE HERE

    # At this point, PPMI is actually PMI, so let's drop all negative values,
    # sparsify, and then compute rank-50 SVD

    # PPMI = max(0, PMI)
    PPMI.data[PPMI.data < 0] = 0
    # sparisfy
    PPMI.eliminate_zeros()

    print('Computing SVD...', file=sys.stderr)
    u, s, vt = svds(PPMI, k = 50)

    p = 1.0
    emb = u * (s ** p)

    # normalize embeddings to unit length so cos(x, y) == x.T * y
    emb = (emb.T / norm(emb, axis=1, ord=2)).T

    print('Saving embeddings...', file=sys.stderr)
    np.save('svd_embeddings.npy', emb)

    print('Done!', file=sys.stderr)
