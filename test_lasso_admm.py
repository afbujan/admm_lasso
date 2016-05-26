from __future__ import division
import pdb,h5py,os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from numpy.linalg import norm
import lasso_admm as ladmm
reload(ladmm)

from sparse_randn import sprandn

"""
Author  : Alex Bujan (adapted from http://www.stanford.edu/~boyd)
Date    : 12/06/2015
"""

np.random.seed(1234)

m = 1500        # number of examples
n = 5000        # number of features
p = 100/n       # sparsity density

x0  = sprandn(n,1,p)
X   = np.random.randn(m,n)
D   = sparse.diags(1./np.sqrt(np.sum(X**2,0)).T,0,(n,n))
X   = D.T.dot(X.T).T
y   = x0.T.dot(X.T).T+np.sqrt(1e-3)*np.random.randn(m,1)

if not os.path.exists('test_lasso_admm.h5'):
    with h5py.File('test_lasso_admm.h5','w') as f:
        g = f.create_group('data')
        g.create_dataset(name='X' ,data=X,compression="gzip")
        g.create_dataset(name='y' ,data=y,compression="gzip")
        g.create_dataset(name='x0',data=x0.todense(),compression="gzip")


alpha_max = norm(X.T.dot(y),np.inf)
alpha = .1*alpha_max

print '\n*Alpha: %.4f'%alpha

x, h = ladmm.lasso_admm(X,y,alpha,1.,1.)

K = len(h['objval'][np.where(h['objval']!=0)])

fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.plot(np.arange(K), h['objval'][:K],'k',ms=10,lw=2)
ax.set_ylabel('f(x^k) + g(z^k)')
ax.set_xlabel('iter (k)')

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(211)
ax1.semilogy(np.arange(K),np.maximum(1e-8,h['r_norm'][:K]),'k',lw=2)
ax1.semilogy(np.arange(K),h['eps_pri'][:K],'k--',lw=2)
ax1.set_ylabel('||r||_2')

ax2 = fig2.add_subplot(212)
ax2.semilogy(np.arange(K),np.maximum(1e-8,h['s_norm'][:K]),'k',lw=2)
ax2.semilogy(np.arange(K),h['eps_dual'][:K],'k--',lw=2)
ax2.set_ylabel('||s||_2')
ax2.set_xlabel('iter (k)')

plt.show()
