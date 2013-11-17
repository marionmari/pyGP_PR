#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGP_PR.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 20/05/2013
#================================================================================

from numpy import diag, dot, maximum
from numpy.linalg import eig, cholesky
from numpy.linalg.linalg import LinAlgError

def nearPD(A):
    tol = 1.e-8
    # Returns the "closest" (up to tol) symmetric positive definite matrix to A.
    # Returns A if it is already Symmetric positive Definite
    count = 0; BOUND = 100
    M = A.copy()
    while count < BOUND:
        M = (M+M.T)/2.
        eigval, Q = eig(M)
        eigval = eigval.real
        Q = Q.real
        xdiag = diag(maximum(eigval, tol))
        M = dot(Q,dot(xdiag,Q.T))
        try:
            L = cholesky(M)
            break
        except LinAlgError:
            count += 1
    if count == BOUND:
        raise Exception("This matrix caused the nearPD algorithm to not converge")
    return M

if __name__ == '__main__':
    from numpy import array
    from numpy.random import random
    from numpy.linalg import norm

    A = array([[2.,-1,0,0.],[-1.,2.,-1,0],[0.,-1.,2.,-1.],[0.,0.,-1.,2.]])
    A = random((4,4))
    M = nearPD(A)
    try:
        L = cholesky(M)
    except LinAlgError:
        print "This shouldn't happen"
    print norm(M-A,'fro')
