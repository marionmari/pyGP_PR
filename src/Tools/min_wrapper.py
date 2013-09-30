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
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 20/05/2013
#================================================================================

# @author: Daniel Marthaler (Fall 2012)
#          update by Marion Neumann (May 20013)
    
import numpy as np
from copy import deepcopy
from scipy.optimize import fmin_bfgs as bfgs
from scipy.optimize import fmin_cg as cg
from scg import scg
from minimize import run
from GPR.UTIL.utils import convert_to_array, convert_to_class


def min_wrapper(hyp, F, Flag, *varargin):
    # Utilize scipy.optimize functions, sgc.py, or minimize.py to
    # minimize the negative log marginal liklihood.
    
    x = convert_to_array(hyp)   # convert the hyperparameter class to an array

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

    elif Flag == 'SCG':
        # use sgc.py
        aa   = scg(x, nlml, dnlml, (F,hyp,varargin), niters = 100)
        hyp  = convert_to_class(aa[0],hyp)
        fopt = aa[1][-1]
        gopt = dnlml(aa[0],F,hyp,varargin)
        return hyp, fopt, gopt, len(aa[1])

    elif Flag == 'Minimize':
        # use minimize.py
        aa   = run(x, nlml, dnlml, (F,hyp,varargin), maxnumfuneval=-100)
        hyp  = convert_to_class(aa[0],hyp)
        fopt = aa[1][-1]
        gopt = dnlml(aa[0],F,hyp,varargin)
        return hyp, fopt, gopt, len(aa[1])

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
