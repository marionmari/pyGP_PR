#===============================================================================
#    Copyright (C) 2013
#    Marion Neumann [marion dot neumann at uni-bonn dot de]
#    Daniel Marthaler [marthaler at ge dot com]
#    Shan Huang [shan dot huang at iais dot fraunhofer dot de]
#    Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
# 
#    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
# 
#    This file is part of pyGPs.
# 
#    pyGPs is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
# 
#    pyGPs is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================

import numpy as np

def nearPD(A):
    tol = 1.e-8
    # Returns the "closest" (up to tol) symmetric positive definite matrix to A.
    # Returns A if it is already Symmetric positive Definite
    count = 0; BOUND = 100
    M = A.copy()
    while count < BOUND:
        M = (M+M.T)/2.
        eigval, Q = np.linalg.eig(M)
        eigval = eigval.real
        Q = Q.real
        xdiag = np.diag(np.maximum(eigval, tol))
        M = np.dot(Q,np.dot(xdiag,Q.T))
        try:
            L = np.linalg.cholesky(M)
            break
        except np.linalg.linalg.LinAlgError:
            count += 1
    if count == BOUND:
        raise Exception("This matrix caused the nearPD algorithm to not converge")
    return M

if __name__ == '__main__':
    A = np.array([[2.,-1,0,0.],[-1.,2.,-1,0],[0.,-1.,2.,-1.],[0.,0.,-1.,2.]])
    A = np.random.random((4,4))
    M = nearPD(A)
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.linalg.LinAlgError:
        print "This shouldn't happen"
    print np.linalg.norm(M-A,'fro')
