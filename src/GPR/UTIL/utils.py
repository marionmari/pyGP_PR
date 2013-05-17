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
    # Created on 08/09/2009
    #
    # @author: Marion Neumann (Aug 2009)
    # Substantial updates by Daniel Marthaler Fall 2012.
    #
    # Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013
    
    
import numpy as np
import matplotlib.pyplot as plt
import Tools.general
from copy import deepcopy
from random import sample

class hyperParameters:
    def __init__(self):
        self.mean = np.array([])
        self.cov  = np.array([])
        self.lik  = np.array([])
        
class dnlzStruct:
    def __init__(self, hyp):
        self.mean = self.cov = self.lik = []
        if hyp.mean != None:
            self.mean = np.zeros(hyp.mean.shape)
        if hyp.cov != None:
            self.cov = np.zeros(hyp.cov.shape) 
        if hyp.lik != None:
            self.lik = np.zeros(hyp.lik.shape)


def cholupdate(R,x,sgn='+'):
    # Placeholder for a python version of MATLAB's cholupdate.  Now it is O(n^3)
    if len(x.shape) == 1:
        # Reshape x so that the dot product below is correct
        x = np.reshape(x,(x.shape[0],1))
    assert(R.shape[0] == x.shape[0])
    A = np.dot(R.T,R)
    if sgn == '+':
        R1 = A + np.dot(x,x.T)
    elif sgn == '-':
        R1 = A - np.dot(x,x.T)
    else:
        raise Exception('Sign needs to be + or - in cholupdate')
    return np.linalg.cholesky(R1).T

def randperm(k):
    # return a random permutation of range(k)
    z = range(k)
    y = []
    ii = 0
    while z and ii < 2*k:
        n = sample(z,1)[0]
        y.append(n)
        z.remove(n)
        ii += 1
    return y

def cool_plotter(xs,ym,ys2,x,y,axisvals=None,file=None):
    xss  = np.reshape(xs,(xs.shape[0],))
    ymm  = np.reshape(ym,(ym.shape[0],))
    ys22 = np.reshape(ys2,(ys2.shape[0],))
    fig = plt.figure()
    ax = fig.add_subplot(111,axisbg=[0.,1.0,0.0,0.2])
    #plt.plot(xs, ym, 'g-', x, y, 'r+', linewidth = 3.0, markersize = 10.0)
    plt.plot(xs, ym, 'r-', x, y, 'y^', linewidth = 3.0, markersize = 5.0)
    plt.fill_between(xss,ymm + 1.*np.sqrt(ys22), ymm - 1.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
    plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.6],linewidths=0.0)
    plt.fill_between(xss,ymm + 3.*np.sqrt(ys22), ymm - 3.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.4],linewidths=0.0)
    plt.grid()
    if axisvals:
        plt.axis(axisvals)
    #plt.xlabel('input x')
    #plt.ylabel('output y')

    #Create figure; attempt to set background to light gray
    fig.set_facecolor("#000000")
    #plt.savefig('GP.png')
    if file and isinstance(file,str):
        plt.savefig(file)
    plt.show()

def plotter(xs,ym,ys2,x,y,axisvals=None,file=None):
    xss  = np.reshape(xs,(xs.shape[0],))
    ymm  = np.reshape(ym,(ym.shape[0],))
    ys22 = np.reshape(ys2,(ys2.shape[0],))
    plt.plot(xs, ym, 'g-', x, y, 'r+', linewidth = 3.0, markersize = 10.0)
    plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
    plt.grid()
    if axisvals:
        plt.axis(axisvals)
    plt.xlabel('input x')
    plt.ylabel('output y')

    if file and isinstance(file,str):
        plt.savefig(file)
    plt.show()

def FITCplotter(u,xs,ym,ys2,x,y,axisvals=None,file=None):
    xss  = np.reshape(xs,(xs.shape[0],))
    ymm  = np.reshape(ym,(ym.shape[0],))
    ys22 = np.reshape(ys2,(ys2.shape[0],))
    plt.plot(xs, ym, 'g-', x, y, 'r+', linewidth = 3.0, markersize = 10.0)
    plt.fill_between(xss,ymm + 2.*np.sqrt(ys22), ymm - 2.*np.sqrt(ys22), facecolor=[0.,1.0,0.0,0.8],linewidths=0.0)
    plt.grid()
    if axisvals:
        plt.axis(axisvals)
    plt.xlabel('input x')
    plt.ylabel('output y')
    plt.plot(u,np.ones_like(u),'kx',markersize=12)
    if file and isinstance(file,str):
        plt.savefig(file)
    plt.show()

def convert_to_array(hyp):
    y = np.concatenate((np.concatenate((hyp.mean, hyp.cov),axis=0),hyp.lik),axis=0)
    return y

def convert_to_class(x,hyp):
    y  = deepcopy(hyp)
    Lm = len(hyp.mean)
    Lc = len(hyp.cov)
    y.mean = x[:Lm]
    y.cov  = x[Lm:(Lm+Lc)]
    y.lik  = x[(Lm+Lc):]
    return y

def unique(x):
    # First flatten x
    y = [item for sublist in x for item in sublist]
    if isinstance(x,np.ndarray):
        n,D = x.shape
        assert(D == 1)
        y = np.array( list(set(x[:,0])) )
        y = np.reshape(y, (len(y),1))
    else:
        y = list(set(y))
    return y

def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def checkParameters(func,hyp,D):
    valt = flatten(Tools.general.feval(func))
    val = 0
    if isinstance(valt,list):
        for v in valt:
            if isinstance(v,str):
                val += eval(v)
            else:
                val += v
    else:
        val = valt
    res = (val == len(hyp))
    return res
