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
#    Marion Neumann, Daniel Marthaler, Shan Huang & Kristian Kersting, 30/09/2013
#================================================================================

import numpy as np
from src.Core.gp import gp
from src.Tools.min_wrapper import min_wrapper 
from src.Tools.utils import convert_to_array, hyperParameters
import matplotlib.pyplot as plt

def HousingPlotter(y,ym,ys2,yt):
    x = range(len(y),len(y)+len(ys))

    plt.plot(y, 'r.', linewidth = 3.0)
    plt.plot(x,ym,'g-', linewidth = 3.0)
    #plt.fill_between(x, ym + 1.*np.sqrt(ys2), ym - 1.*np.sqrt(ys2), facecolor=[0.,1.0,0.0,0.9],linewidths=0.0)
    #plt.fill_between(x, ym + 2.*np.sqrt(ys2), ym - 2.*np.sqrt(ys2), facecolor=[0.,1.0,0.0,0.7],linewidths=0.0)
    #plt.fill_between(x, ym + 3.*np.sqrt(ys2), ym - 3.*np.sqrt(ys2), facecolor=[0.,1.0,0.0,0.5],linewidths=0.0)
    
    plt.plot(x,ys, 'bx', linewidth = 3.0, markersize = 5.0)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Median Home Values')
    plt.show()    

if __name__ == '__main__':
    infile = '../../data/housing.txt'
    data = np.genfromtxt(infile)

    DN, DD = data.shape
    N = 150
    # Get all data (exclude the 4th column which is binary) except the last 50 points for training
    x  = np.concatenate((data[:-N,:4],data[:-N,5:-1]),axis=1)
    # The function we will perform regression on:  Median Value of owner occupied homes
    y  = np.reshape(data[:-N,-1],(len(data[:-N,-1]),1))
    # Test on the last 50 points
    xs  = np.concatenate((data[-N:,:4],data[-N:,5:-1]),axis=1)
    ys = np.reshape(data[-N:,-1],(N,1))
    N,D = x.shape
    ## DEFINE parameterized covariance function
    covfunc = [ ['kernels.covSum'], [ ['kernels.covSEiso'],['kernels.covNoise'] ] ]

    ## DEFINE parameterized mean function
    meanfunc = [ ['means.meanZero'] ]      

    ## DEFINE parameterized inference and liklihood functions
    inffunc = ['inferences.infExact']
    likfunc = ['likelihoods.likGauss']

    ## SET (hyper)parameters
    hyp = hyperParameters()

    ## SET (hyper)parameters for covariance and mean
    hyp.cov = np.array([np.log(1.), np.log(1.), np.log(1.)])
    hyp.mean = np.array([])

    hyp.lik = np.array([np.log(0.1)])

    print 'Initial mean = ',hyp.mean
    print 'Initial covariance = ',hyp.cov
    print 'Initial liklihood = ',hyp.lik

    [nlml, post] = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)
    print 'Initial negative log marginal likelihood = ',nlml
    
    ##----------------------------------------------------------##
    ## STANDARD GP (prediction)                                 ##
    ##----------------------------------------------------------## 
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xs)
    ym = vargout[0]; ys2 = vargout[1]
    m  = vargout[2]; s2  = vargout[3]
    
    HousingPlotter(y,ym,ys2,ys)
    ##----------------------------------------------------------##
    ## STANDARD GP (training)                                   ##
    ## OPTIMIZE HYPERPARAMETERS                                 ##
    ##----------------------------------------------------------##
    ## -> parameter training using (off the shelf) conjugent gradient (CG) optimization (NOTE: SCG is faster)
    from time import clock
    t0 = clock()
    vargout = min_wrapper(hyp,gp,'SCG',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    t1 = clock()
    hyp = vargout[0]

    #vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xs)
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,np.concatenate((data[:,:4],data[:,5:-1])))
    ym = vargout[0]; ys2 = vargout[1]
    m  = vargout[2]; s2  = vargout[3]

    print 'Time to optimize = ',t1-t0
    print 'Optimized mean = ',hyp.mean
    print 'Optimized covariance = ',hyp.cov
    print 'Optimized liklihood = ',hyp.lik

    [nlml, post] = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)
    print 'Final negative log marginal likelihood = ',nlml

    HousingPlotter(y,ym,ys2,ys)
