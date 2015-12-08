from __future__ import division
import pdb,time,h5py,os
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm,cholesky
from mpi4py import MPI
from optparse import OptionParser

"""
Author  : Alex Bujan (adapted from http://www.stanford.edu/~boyd)
Date    : 12/06/2015
"""

def main():
    usage = '%prog [options]'
    parser = OptionParser(usage)
    parser.add_option("-o",type="string",default=os.getcwd(),\
        help="hdf5 file to store the results")
    parser.add_option("-i",type="string",default='',\
        help="hdf5 file containing input data (X,y pairs)")
    parser.add_option("--alpha",type="float",default=.5,\
        help="scalar regularization parameter for Lasso")
    parser.add_option("--rho",type="float",default=1.,\
        help="augmented Lagrangian parameter")
    parser.add_option("--max_iter",type="int",default=50,\
        help="max number of ADMM iterations")
    parser.add_option("--abs_tol",type="float",default=1e-3,\
        help="absolute tolerance for early stopping")
    parser.add_option("--rel_tol",type="float",default=1e-2,\
        help="relative tolerance for early stopping")
    parser.add_option("--verbose",action='store_true',\
        dest='verbose',help="print information in the terminal")
    parser.add_option("--debug",action='store_true',\
        dest='debug',help="print information in the terminal")

    (options, args) = parser.parse_args()

    if options.verbose:
        verbose=True
    else:
        verbose=False

    if options.debug:
        debug=True
    else:
        debug=False

    lasso_admm(inputFile=options.i,outputFile=options.o,\
               alpha=options.alpha,rho=options.rho,verbose=verbose,\
               max_iter=options.max_iter,abs_tol=options.abs_tol,\
               rel_tol=options.rel_tol,debug=debug)

def lasso_admm(inputFile,outputFile,alpha=.5,rho=1.,verbose=True,\
                max_iter=50,abs_tol=1e-3,rel_tol=1e-2,debug=False):
    """
     Solve lasso problem via ADMM
    -----------------------------
     Lasso problem:

       minimize 1/2*|| Ax - y ||_2^2 + alpha || x ||_1

    Input:
        - hdf5 input file with following contents:
            - data/X : design matrix (samples,features)
            - data/y : target variable (samples)
    Output:
        - hdf5 output file with following values:
            - x      : solution of the Lasso problem (weights)
            - objval : objective value
            - r_norm : primal residual norm
            - s_norm : dual residual norm
            - eps_pri: tolerance for primal residual norm
            - eps_pri: tolerance for dual residual norm
    """

    '''
    MPI
    '''
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    N = size

    '''
    Data
    '''
    #TODO: improve with parallel hdf5
    if rank==0:
        with h5py.File(inputFile,'r') as f:
            X   = np.copy(f['data/X'].value)
            y   = np.copy(f['data/y'].value)
            m,n = f['data/X'].shape
    else:
        n  = np.zeros(1).astype('int')
        m  = np.zeros(1).astype('int')

    n = comm.bcast(n,root=0)
    m = comm.bcast(m,root=0)

    if rank!=0:
        X = np.empty((m,n),dtype=np.float64)
        y = np.empty(m,dtype=np.float64)

    '''
    Broadcast data
    '''
    comm.Barrier()
    comm.Bcast([X,MPI.DOUBLE])
    comm.Bcast([y,MPI.DOUBLE])

    '''
    Select sample block
    '''
    X   = X[rank::N,:]
    m,n = X.shape
    y   = y.ravel()[rank::N].reshape((m,1))

    if rank==0:
        tic = time.time()

    #save a matrix-vector multiply
    Xty = X.T.dot(y)
    
    if verbose and debug:
        print "\n*Process #: %i \t*X   size: %s"%(rank,str(X.shape))
        print "\n*Process #: %i \t*Xty size: %s"%(rank,str(Xty.shape))

    #initialize ADMM solver
    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))
    r = np.zeros((n,1))

    send = np.zeros(3)
    recv = np.zeros(3)

    # cache the (Cholesky) factorization
    L,U = factor(X,rho)

    # Saving state
    if rank==0 and verbose:
        print '\n%3s\t%10s\t%10s\t%10s\t%10s\t%10s' %('iter',
                                                      'r norm', 
                                                      'eps pri', 
                                                      's norm', 
                                                      'eps dual', 
                                                      'objective')
    objval     = []
    r_norm     = []
    s_norm     = []
    eps_pri    = []
    eps_dual   = []

    '''
    ADMM solver loop
    '''
    for k in xrange(max_iter):

        # u-update
        if k!=0:
            u+=(x-z)

        # x-update 
        q = Xty+rho*(z-u) #(temporary value)

        if m>=n:
            x = spsolve(U,spsolve(L,q))[...,np.newaxis]
        else:
            ULXq = spsolve(U,spsolve(L,X.dot(q)))[...,np.newaxis]
            x = (q*1./rho)-((X.T.dot(ULXq))*1./(rho**2))

        w = x + u

        send[0] = r.T.dot(r)[0][0]
        send[1] = x.T.dot(x)[0][0]
        send[2] = u.T.dot(u)[0][0]/(rho**2)

        zprev = np.copy(z)

        comm.Barrier()
        comm.Allreduce([w,MPI.DOUBLE],[z,MPI.DOUBLE])
        comm.Allreduce([send,MPI.DOUBLE],[recv,MPI.DOUBLE])

        # z-update
        z = soft_threshold(z*1./N,alpha*1./(N*rho))

        # diagnostics, reporting, termination checks
        objval.append(objective(X,y,alpha,x,z))
        #prires -> norm(x-z)
        r_norm.append(np.sqrt(recv[0]))
        #dualres -> norm(-rho*(z-zold))
        s_norm.append(np.sqrt(N)*rho*norm(z-zprev))
        eps_pri.append(np.sqrt(n*N)*abs_tol+\
                       rel_tol*np.maximum(np.sqrt(recv[1]),np.sqrt(N)*norm(z)))
        eps_dual.append(np.sqrt(n*N)*abs_tol+rel_tol*np.sqrt(recv[2]))

        if rank==0 and verbose:
            print '%4d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f' %(k+1,\
                                                              r_norm[k],\
                                                              eps_pri[k],\
                                                              s_norm[k],\
                                                              eps_dual[k],\
                                                              objval[k])

        if r_norm[k]<eps_pri[k] and s_norm[k]<eps_dual[k] and k>0:
            break

        #Compute residual
        r = x-z

    if rank==0:
        toc = time.time()-tic
        if verbose:
            print "\nElapsed time is %.2f seconds"%toc
        '''
        Store results
        '''
        with h5py.File(outputFile,'w') as f:
            f.attrs['time']     = toc
            f.attrs['alpha']    = alpha
            f.attrs['rho']      = rho
            f.attrs['abs_tol']  = abs_tol
            f.attrs['rel_tol']  = rel_tol
            f.create_dataset(name='z',data=z.ravel(),compression='gzip')
            f.create_dataset(name='objval',data=np.asarray(objval),\
                             compression='gzip')
            f.create_dataset(name='r_norm',data=np.asarray(r_norm),\
                             compression='gzip')
            f.create_dataset(name='s_norm',data=np.asarray(s_norm),\
                             compression='gzip')
            f.create_dataset(name='eps_pri',data=np.asarray(eps_pri),\
                             compression='gzip')
            f.create_dataset(name='eps_dual',data=np.asarray(eps_dual),\
                             compression='gzip')

def objective(X,y,alpha,x,z):
    return .5*np.square(X.dot(x)-y).sum()+alpha*norm(z,1)

def factor(X,rho):
    m,n = X.shape
    if m>=n:
       L = cholesky(X.T.dot(X)+rho*sparse.eye(n))
    else:
       L = cholesky(sparse.eye(m)+1./rho*(X.dot(X.T)))
    L = sparse.csc_matrix(L)
    U = sparse.csc_matrix(L.T)
    return L,U

def soft_threshold(v,k):
    v[np.where(v>k)]-=k
    v[np.where(v<-k)]+=k
    v[np.intersect1d(np.where(v>-k),np.where(v<k))] = 0
    return v

if __name__=='__main__':
    main()
