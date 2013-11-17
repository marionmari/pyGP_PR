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

from src.Core.gp import gp
from src.Tools.solve_chol import solve_chol
import src.Tools.general
from src.Tools.min_wrapper import min_wrapper
import src.Tools.nearPD
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from src.Tools.utils import *
from scipy.optimize import fmin_cg as cg

if __name__ == '__main__':
    PLOT = True

    ## LOAD data
    demoData = np.load('../../data/classification_data.npz')
    x = demoData['x']            # training data
    y = demoData['y']            # training target
    xstar = demoData['xstar']    # test data
    n = xstar.shape[0]           # number of test points

    ## DATA only needed in plotting
    if PLOT:  
        x1 = demoData['x1'] 
        x2 = demoData['x2']                    
        t1 = demoData['t1'] 
        t2 = demoData['t2'] 
        p1 = demoData['p1']  
        p2 = demoData['p2']  

    ### PLOT data 
    if PLOT:
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        pc = plt.contour(t1, t2, np.reshape(p2/(p1+p2), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        plt.axis([-4, 4, -4, 4])
        plt.show()
    
    ## DEFINE parameterized mean and covariance functions
    meanfunc = ['means.meanConst']  
    covfunc  = ['kernels.covSEard'] 
    ## DEFINE likelihood function used 
    likfunc = ['likelihoods.likErf']
    ## SPECIFY inference method
    inffunc = ['inferences.infLaplace']

    ## SET (hyper)parameters
    hyp = hyperParameters()
    hyp.mean = np.array([-2.842117459073954])
    hyp.cov  = np.array([0.051885508906388,0.170633324977413,1.218386482861781])
    
    ##----------------------------------------------------------##
    ## STANDARD GP (example 1)                                  ##
    ##----------------------------------------------------------##
    print '...example 1: prediction...'
    ## GET negative log marginal likelihood
    [nlml,dnlZ,post] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, None, None, True)
    print "nlml = ", nlml    
    
    ## PREDICTION
    [ymu,ys2,fmu,fs2,lp,post] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xstar, np.ones((n,1)) )
    
    ## PLOT log predictive probabilities
    if PLOT:
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        plt.axis([-4, 4, -4, 4])
        plt.show()


    ##----------------------------------------------------------##
    ## SPARSE GP (example 2)                                    ##
    ##----------------------------------------------------------##   
    print '...example 2: FITC training and prediction...'
    ## SPECIFY inducing points
    u1,u2 = np.meshgrid(np.linspace(-2,2,5),np.linspace(-2,2,5))
    u = np.array(zip(np.reshape(u2,(np.prod(u2.shape),)),np.reshape(u1,(np.prod(u1.shape),)))) 
    del u1, u2
    nu = u.shape[0]

    ## SPECIFY FITC covariance function
    covfuncF = [['kernels.covFITC'], covfunc, u]  

    ## SPECIFY FITC inference method      
    inffunc = ['inferences.infFITC_EP'] 

    ## GET negative log marginal likelihood
    [nlml,dnlZ,post] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, None, None, True)
    print "nlml =", nlml   

    ## TRAINING: OPTIMIZE hyperparameters
    [hyp_opt, fopt, gopt, funcCalls] = min_wrapper(hyp,gp,'Minimize',inffunc,meanfunc,covfuncF,likfunc,x,y,None,None,True)  # minimize by Carl Rasmussen
    print 'Optimal nlml =', fopt

    ## FITC PREDICTION     
    [ymu,ys2,fmu,fs2,lp,post] = gp(hyp_opt, inffunc, meanfunc, covfuncF, likfunc, x, y, xstar, np.ones((n,1)) )

    ## PLOT log predictive probabilities
    if PLOT:
        fig = plt.figure()
        plt.plot(x1[:,0], x1[:,1], 'b+', markersize = 12)
        plt.plot(x2[:,0], x2[:,1], 'r+', markersize = 12)
        plt.plot(u[:,0],u[:,1],'ko', markersize=12)
        pc = plt.contour(t1, t2, np.reshape(np.exp(lp), (t1.shape[0],t1.shape[1]) ))
        fig.colorbar(pc)
        plt.grid()
        plt.axis([-4, 4, -4, 4])
        plt.show()
        
