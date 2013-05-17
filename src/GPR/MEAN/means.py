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

import Tools
import numpy as np
import math


def meanMask(meanfunc, meanhyper=None, x=None, der=None):
    ''' meanMask - compose a mean function as another mean
        function (meanfunc), but with only a subset of dimensions of x. 
        '''
    
    mask = meanfunc[0]              # The indicies to be masked (should be a list of integers)
    mean = meanfunc[1]              # covariance function to $
    
    if meanhyper == None:           # report number of parameters
        A = Tools.general.feval(meanfunc[1])
        return [A]
    
    n, D = x.shape
    
    assert(len(mask) <= D)
    assert(max(mask) <= D)
    assert(min(mask) >= 0)
    
    if der == None:                 # compute covariance matix for dataset x
        A = Tools.general.feval(mean, meanhyper, x[:,mask])
    else:                           # compute derivatives
        A = Tools.general.feval(mean, meanhyper, x[:,mask], der)
    
    return A

def meanConst(meanhyper=None, x=None, der=None):
    ''' 
    Constant mean function. The mean function is parameterized as:
      m(x) = c
    
    The hyperparameter is:
      meanhyper = [ c ]
    '''
    if meanhyper == None:                       # report number of parameters
        return [1]

    n,D = x.shape
    c = meanhyper[0]
 
    if der == None:                             # evaluate mean
        A = c*np.ones((n,1)) 
    elif isinstance(der, int) and der == 0:     # compute derivative vector wrt c
        A = np.ones((n,1)) 
    else:   
        A = np.zeros((n,1)) 
    return A

def meanLinear(meanhyper=None, x=None, der=None):
    ''' 
    Linear mean function. The mean function is parameterized as:
      m(x) = sum_i ai * x_i 
    
    The hyperparameter is:
      meanhyper = [ a_1 
                    a_2
                    ...
                    a_D ]
    '''

    if meanhyper == None:                       # report number of parameters
        return ['D']
    
    n, D = x.shape
    c = np.array(meanhyper)
    c = np.reshape(c,(len(c),1))

    if der == None:                             # evaluate mean
        A = np.dot(x,c)
    elif isinstance(der, int) and der < D:      # compute derivative vector wrt meanparameters
        A = np.reshape(x[:,der], (len(x[:,der]),1) ) 
    else:   
        A = np.zeros((n,1)) 
    return A

def meanOne(meanhyper=None, x=None, der=None):
    ''' 
    One mean function. The mean function does not have any parameters
      m(x) = 1
    
    '''
    if meanhyper == None:                       # report number of parameters
        return [0]

    n,D = x.shape

    if der == None:                             # evaluate mean
        A = np.ones((n,1)) 
    else:   
        A = np.zeros((n,1))
    return A

def meanZero(meanhyper=None, x=None, der=None):
    ''' 
    Zero mean function. The mean function does not have any parameters
      m(x) = 1
    
    '''
    if meanhyper == None:                       # report number of parameters
        return [0]
    n, D = x.shape
    A = np.zeros((n,1)) 
    return A

def meanProd(meanfunc, meanhyper=None, x=None, der=None):
    ''' meanProd - compose a mean function as the product of other mean
     functions. This function doesn't actually compute very much on its own, it
     merely does some bookkeeping, and calls other mean functions to do the
     actual work. 
    '''
    
    def DetermineNumberOfParameters(v,no_param):
        if isinstance(no_param, int):
            v.append(no_param)
        elif isinstance(no_param,str):          # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D': temp = int(D)
            if pram_str[1]=='+': temp += int(pram_str[2])
            elif pram_str[1]=='-': temp -= int(pram_str[2])
            else:
                raise Exception(["Error: number of parameters of "+meanfunc[i] +" unknown!"])
            v.append(temp)
        elif isinstance(no_param, list):
            # The number of hyperparameters for this piece of meanfunc is the sum
            # of all of them in this composition
            temp = [0]
            for jj in xrange(len(no_param)):
                DetermineNumberOfParameters(temp,no_param[jj])
            v.append(sum(temp))
        else:
            # This is an error, we should never be here
            raise Exception("Error in return of number of parameters")
        return v
    
    if meanhyper == None:    # report number of parameters
        A = [Tools.general.feval(meanfunc[0])]
        for ii in range(1,len(meanfunc)):
            A.append(Tools.general.feval(meanfunc[ii]))
        return A

    [n, D] = x.shape

    # SET vector v (v indicates how many parameters each meanfunc has 
    # (NOTE : v[i]=number of parameters + 1 -> this is because of the indexing of python!))
    v = [0]
    for ii in range(1,len(meanfunc)+1):
        no_param = Tools.general.feval(meanfunc[ii-1])
        DetermineNumberOfParameters(v,no_param)

    if der == None:                             # compute mean vector
        A = np.ones((n, 1))                     # allocate space for mean vector
        for ii in range(1,len(meanfunc)+1):     # iteration over multiplicand functions
            f = meanfunc[ii-1]
            B = Tools.general.feval(f,meanhyper[sum(v[0:ii]):sum(v[0:ii])+v[ii]], x)  # accumulate means
            A *= B

    elif isinstance(der, int):                  # compute derivative vector   
        tmp = 0
        A = np.ones((n, 1))                     # allocate space for derivative vector
        flag = True
        for ii in range(1,len(meanfunc)+1):
            tmp += v[ii]
            if der<tmp and flag:
                flag = False
                f = meanfunc[ii-1]              # i: which mean function
                jj = der-(tmp-v[ii])            # j: which parameter in that mean
                # compute derivative
                s = sum(v[0:ii])
                A *= Tools.general.feval(f, meanhyper[s:(s+v[ii])], x, int(jj))
            else:
                f = meanfunc[ii-1]              # i: which mean function
                s = sum(v[0:ii])
                A *= Tools.general.feval(f, meanhyper[s:(s+v[ii])], x)
    else:                            
        A = np.zeros((n,1))
    return A

def meanSum(meanfunc, meanhyper=None, x=None, der=None):
    '''covSum - compose a mean function as the sum of other mean
    functions. This function doesn't actually compute very much on its own, it
    merely does some bookkeeping, and calls other mean functions to do the
    actual work. '''
    
    def DetermineNumberOfParameters(v,no_param):
        if isinstance(no_param, int):
            v.append(no_param)
        elif isinstance(no_param,str):          # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D':
                temp = int(D)
            if len(pram_str)>1:
                if pram_str[1]=='+': temp += int(pram_str[2])
                elif pram_str[1]=='-': temp -= int(pram_str[2])
                else: raise Exception(["Error: number of parameters of "+meanfunc[ii][0] +" unknown!"])
            v.append(temp)
        elif isinstance(no_param, list):
            # The number of hyperparameters for this piece of meanfunc is the sum
            # of all of them in this composition
            temp = [0]
            for jj in xrange(len(no_param)):
                DetermineNumberOfParameters(temp,no_param[jj])
            v.append(sum(temp))
        else:
            # This is an error, we should never be here
            raise Exception("Error in return of number of parameters")
        return v

    if meanhyper == None:    # report number of parameters
        A = [Tools.general.feval(meanfunc[0])]
        for ii in range(1,len(meanfunc)):
            A.append(Tools.general.feval(meanfunc[ii]))
        return A

    [n, D] = x.shape

    # SET vector v (v indicates how many parameters each meanfunc has 
    # (NOTE : v[i]=number of parameters + 1 -> this is because of the indexing of python!))
    
    v = [0]    # needed for technical reasons
    for ii in range(1,len(meanfunc)+1):
        no_param = Tools.general.feval(meanfunc[ii-1])
        DetermineNumberOfParameters(v,no_param)

    if der == None:                             # compute mean vector
        A = np.zeros((n, 1))                    # allocate space for mean vector
        for ii in range(1,len(meanfunc)+1):     # iteration over summand functions
            f = meanfunc[ii-1]
            s = sum(v[0:ii])
            A = A + Tools.general.feval(f, meanhyper[s:(s+v[ii])], x)  # accumulate means

    elif isinstance(der, int):                  # compute derivative vector
        A = np.zeros((n, 1))                    # allocate space for mean vector
        tmp = 0
        for ii in range(1,len(meanfunc)+1):
            tmp += v[ii]
            if der<tmp:
                jj = der-(tmp-v[ii]); break     # j: which parameter in that mean
        f = meanfunc[ii-1]                      # i: which mean function
        # compute derivative
        s = sum(v[0:ii])
        A = A + Tools.general.feval(f, meanhyper[s:(s+v[ii])], x, int(jj))  # accumulate means

    else:                                       # compute test set means
        A = np.zeros((n,1))
    return A

def meanScale(meanfunc, meanhyper=None, x=None, der=None):
    '''Compose a mean function as a scaled version of another one
    k(x^p,x^q) = sf2 * k0(x^p,x^q)
    
    The hyperparameter is :
    
    meanhyper = [ log(sf2) ]

    This function doesn't actually compute very much on its own. it merely does
    some bookkeeping, and calls another mean function to do the actual work.
    '''

    if meanhyper == None:                                   # report number of parameters
        A = [1]
        A.append( Tools.general.feval(meanfunc[0]) )
        return A

    c = meanhyper[0]                                        # scale parameter

    if der == None:                                         # compute mean vector
        f = meanfunc[0]
        A = c * Tools.general.feval(f, meanhyper[1:], x)    # accumulate means

    elif isinstance(der, int) and der == 0:                 # compute derivative w.r.t. c
        f = meanfunc[0]
        A = Tools.general.feval(f, meanhyper[1:], x)

    else:                                   
        f = meanfunc[0]
        A = c * Tools.general.feval(f, meanhyper[1:], x, None, der-1)
    return A

def meanPow(meanfunc, meanhyper=None, x=None, der=None):
    '''Compose a mean function as the power of another one
      m(x) = m0(x) ** d
    
    This function doesn't actually compute very much on its own. it merely does
    some bookkeeping, and calls another mean function to do the actual work.
    '''

    if meanhyper == None:                                       # report number of parameters
        A = [1]
        A.append( Tools.general.feval(meanfunc[0]) )
        return A

    d = np.abs(np.floor(meanhyper[0])) 
    d = max(d,1)

    if der == None:                                             # compute mean vector
        f = meanfunc[0]
        A = ( Tools.general.feval(f, meanhyper[1:], x) )**d     # accumulate means

    else:                                                       # compute derivative vector
        f = meanfunc[0]
        A = d * (Tools.general.feval(f, meanhyper[1:], x))**(d-1) \
                * Tools.general.feval(f, meanhyper[1:], x, None, der-1)
    return A
