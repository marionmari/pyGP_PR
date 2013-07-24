#===============================================================================
# Copyright (C) 2013
# Marion Neumann [marion dot neumann at uni-bonn dot de]
# Daniel Marthaler [marthaler at ge dot com]
# Shan Huang [shan dot huang at iais dot fraunhofer dot de]
# Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
#
# Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
#
# This file is part of pyGPs.
#
# pyGPs is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# pyGPs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================


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
