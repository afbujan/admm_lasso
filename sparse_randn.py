from numpy.random import random_integers
from scipy import rand, randn, ones
from scipy.sparse import csr_matrix

def _rand_sparse(m, n, density):
    # check parameters here
    nnz = max( min( int(m*n*density), m*n), 0)
    row  = random_integers(low=0, high=m-1, size=nnz)
    col  = random_integers(low=0, high=n-1, size=nnz)
    data = ones(nnz, dtype='int8')
    # duplicate (i,j) entries will be summed together
    return csr_matrix( (data,(row,col)),shape=(m,n))

def sprand(m, n, density):
    """Document me"""
    A = _rand_sparse(m, n, density)
    A.data = rand(A.nnz)
    return A

def sprandn(m, n, density):
    """Document me"""
    A = _rand_sparse(m, n, density)
    A.data = randn(A.nnz)
    return A 
