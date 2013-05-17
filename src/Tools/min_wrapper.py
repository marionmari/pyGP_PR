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

    # @author: Daniel Marthaler (Fall 2012)
    #
    # Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013
    
import numpy as np
from copy import deepcopy
from scipy.optimize import fmin_bfgs as bfgs
from scipy.optimize import fmin_cg as cg

from GPR.UTIL.utils import convert_to_array, convert_to_class

def min_wrapper(hyp, F, Flag, *varargin):
    # Utilize scipy.optimize functions to minimize the negative log marginal liklihood.  This is REALLY inefficient!
    x = convert_to_array(hyp) # Converts the hyperparameter class to an array

    if Flag == 'CG':
        aa = cg(nlml, x, dnlml, (F,hyp,varargin), maxiter=100, disp=False, full_output=True)
        x = aa[0]; fopt = aa[1]; funcCalls = aa[2]; gradcalls = aa[3]
        if aa[4] == 1:
            print "Maximum number of iterations exceeded."
        elif aa[4] ==  2:
            print "Gradient and/or function calls not changing."
        gopt = dnlml(x,F,hyp,varargin)
        return convert_to_class(x,hyp), fopt, gopt, funcCalls

    elif Flag == 'BFGS':
        # Use BFGS
        aa = bfgs(nlml, x, dnlml, (F,hyp,varargin), maxiter=100, disp=False, full_output=True)
        x = aa[0]; fopt = aa[1]; gopt = aa[2]; Bopt = aa[3]; funcCalls = aa[4]; gradcalls = aa[5]
        if aa[6] == 1:
            print "Maximum number of iterations exceeded."
        elif aa[6] ==  2:
            print "Gradient and/or function calls not changing."
        if isinstance(fopt, np.ndarray):
            fopt = fopt[0]
        return convert_to_class(x,hyp), fopt, gopt, funcCalls

    else:
        raise Exception('Incorrect usage of optimization flag in min_wrapper')

def nlml(x,F,*varargin):
    hyp = varargin[0]
    temp = list(varargin[1:][0])
    temp[-1] = False

    f = lambda z: F(z,*temp)
    X = convert_to_class(x,hyp)
    vargout = f(X)
    return vargout[0]

def dnlml(x,F,*varargin):
    hyp = varargin[0]
    temp = list(varargin[1:][0])
    temp[-1] = True
    f = lambda z: F(z,*temp)
    X = convert_to_class(x,hyp)
    vargout = f(X)
    z = convert_to_array(vargout[1])
    return z
