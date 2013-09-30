#================================================================================
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at cs dot tu-dortmund dot de]
#
#    This file is part of pyGP_FN.
#    The software package is released under the BSD 2-Clause (FreeBSD) License.
#
#    Copyright (c) by
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 15/01/2010
#================================================================================


import numpy as np

def solve_chol(L, B):
    # solve_chol - solve linear equations from the Cholesky factorization.
    # Solve A*X = B for X, where A is square, symmetric, positive definite. The
    # input to the function is R the Cholesky decomposition of A and the matrix B.
    # Example: X = solve_chol(chol(A),B);
    
    try:
        assert(L.shape[0] == L.shape[1] and L.shape[0] == B.shape[0])
    except AssertionError:
        raise Exception('Wrong sizes of matrix arguments in solve_chol.py');

    X = np.linalg.solve(L,np.linalg.solve(L.T,B))
    return X
