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
import src.Tools.general
from src.Tools.utils import convert_to_array, hyperParameters, plotter

if __name__ == '__main__':

    # Example demo for pyGP prediction of carbon dioxide concentration using 
    # the Mauna Loa CO2 data [Pieter Tans, Aug 2012]. 
    #
    # The data is constantly updated and publically available under the link:
    # ftp://ftp.cmdl.noaa.gov/ccg/co2/trends/co2_mm_mlo.txt
    #
    # The used covariance function was proposed in [Gaussian Processes for 
    # Machine Learning,Carl Edward Rasmussen and Christopher K. I. Williams, 
    # The MIT Press, 2006. ISBN 0-262-18253-X]. 
    # 
    # Copyright (c) by Marion Neumann and Daniel Marthaler, 20/05/2013

    ## LOAD data
    infile = '../../data/mauna.txt'	# Note: Samples with value -99.99 were dropped.
    f      = open(infile,'r')
    year   = []
    co2    = []
    for line in f:
        z  = line.split('  ')
        z1 = z[1].split('\n')
        if float(z1[0]) != -99.99:
            year.append(float(z[0]))
            co2.append(float(z1[0]))

    X  = [i for (i,j) in zip(year,co2) if i < 2004]
    y  = [j for (i,j) in zip(year,co2) if i < 2004]
    xx = [i for (i,j) in zip(year,co2) if i >= 2004]
    yy = [j for (i,j) in zip(year,co2) if i >= 2004]

    x = np.array(X)
    y = np.array(y)
    x = x.reshape((len(x),1))
    y = y.reshape((len(y),1))

    n,D = x.shape
    ## DEFINE parameterized covariance function
    P = D # The dimension of the input
    N = 5
    covfunc = [ ['kernels.covSum'], [[ ['kernels.covProd'], P*[[ ['kernels.covSum'], N*[['kernels.covSpectralSingle']] ]] ], ['kernels.covNoise'] ] ]

    ## DEFINE parameterized mean function
    meanfunc = [ ['means.meanZero'] ]      

    ## DEFINE parameterized inference and liklihood functions
    inffunc = ['inferences.infExact']
    likfunc = ['likelihoods.likGauss']

    ## SET (hyper)parameters
    hyp = hyperParameters()

    ## SET (hyper)parameters for covariance and mean 
    hyp.cov = np.log(np.random.random(4*N*P+1))
    # Now replace the values of hyp.cov that correspond to the indices for P
    for ii in range(P):
        for jj in range(N):
            ind = ii*N*4 + jj*4 + 4
            hyp.cov[ind-1] =  ii

    hyp.mean = np.array([])

    sn = 0.1
    hyp.lik = np.array([np.log(sn)])

    ##----------------------------------------------------------##
    ## STANDARD GP (prediction)                                 ##
    ##----------------------------------------------------------## 
    xs = np.arange(2004+1./24.,2024-1./24.,1./12.)     # TEST POINTS
    xs = xs.reshape(len(xs),1)

    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xs)
    ym = vargout[0]; ys2 = vargout[1]
    m  = vargout[2]; s2  = vargout[3]
    plotter(xs,ym,ys2,x,y)#,[1955, 2030, 310, 420])
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
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xs)
    ym = vargout[0]; ys2 = vargout[1]
    m  = vargout[2]; s2  = vargout[3]

    print 'Time to optimize = ',t1-t0
    print 'Optimized mean = ',hyp.mean
    print 'Optimized covariance = ',hyp.cov
    print 'Optimized liklihood = ',hyp.lik
    
    plotter(xs,ym,ys2,x,y,[1955, 2030, 310, 420])

