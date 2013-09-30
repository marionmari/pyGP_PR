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
import matplotlib.pyplot as plt
from gp import gp
from Tools.min_wrapper import min_wrapper
import Tools.general
from UTIL.utils import convert_to_array, hyperParameters, plotter, FITCplotter
from time import clock

if __name__ == '__main__':
    PLOT = True

    ## LOAD DATA
    demoData = np.load('../../data/regression_data.npz')
    x = demoData['x']            # training data
    y = demoData['y']            # training target
    xstar = demoData['xstar']    # test data

    ## PLOT data
    if PLOT:
        plt.plot(x,y,'r+',markersize=12)
        plt.axis([-1.9,1.9,-0.9,3.9])
        plt.grid()
        plt.xlabel('input x')
        plt.ylabel('output y')
        plt.show()         
    
    ## DEFINE parameterized mean and covariance functions
    covfunc  = [['kernels.covPoly']]
    meanfunc = [ ['means.meanSum'], [ ['means.meanLinear'] , ['means.meanConst'] ] ]
    ## DEFINE likelihood function used
    likfunc  = ['lik.likGauss']
    ## SPECIFY inference method
    inffunc  = ['inf.infExact']
    
    ## SET (hyper)parameters
    hyp = hyperParameters()
    hyp.cov = np.array([np.log(0.25),np.log(1.0),1.0])
    hyp.mean = np.array([0.5,1.0])
    hyp.lik = np.array([np.log(0.1)])
    
    ##----------------------------------------------------------##
    ## STANDARD GP (example 1)                                  ##
    ##----------------------------------------------------------##
    print '...example 1: prediction...'
    ## PREDICTION
    t0 = clock()
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    t1 = clock()
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2 = vargout[3]
    
    print 'Time for prediction =',t1-t0
    
    ## PLOT results
    if PLOT:
        plotter(xstar,ym,s2,x,y,[-2, 2, -0.9, 3.9])

    ## GET negative log marginal likelihood
    [nlml, post] = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)
    print "nlml =", nlml


    ##----------------------------------------------------------##
    ## STANDARD GP (example 2)                                  ##
    ##----------------------------------------------------------##
    print '...example 2: prediction...'
    ## USE another covariance function	-> for use of composite covariance functions see demoMaunaLoa.py
    covfunc = [ ['kernels.covSEiso'] ]
    
    ### SET (hyper)parameters
    hyp2 = hyperParameters()
    hyp2.cov = np.array([-1.0,0.0])
    hyp2.mean = np.array([0.5,1.0])
    hyp2.lik = np.array([np.log(0.1)])

    ## PREDICTION
    t0 = clock()
    vargout = gp(hyp2,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    t1 = clock()
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2 = vargout[3]
    
    print 'Time for prediction =',t1-t0
    
    ## PLOT results
    if PLOT:
        plotter(xstar,ym,ys2,x,y,[-2, 2, -0.9, 3.9])
    
    ## GET negative log marginal likelihood
    [nlml, post] = gp(hyp2,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)
    print "nlml =", nlml


    ##----------------------------------------------------------##
    ## STANDARD GP (example 3)                                  ##
    ##----------------------------------------------------------##
    print '...example 3: training and prediction...'
    ## TRAINING: OPTIMIZE HYPERPARAMETERS      
    ## -> parameter training via off-the-shelf optimization   
    t0 = clock()
    [hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2,gp,'Minimize',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True) # minimize
    #[hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2,gp,'CG',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)	    # conjugent gradient
    #[hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2,gp,'SCG',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)	    # scaled conjugent gradient (faster than CG) 
    #[hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2,gp,'BFGS',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)    # quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS)
    t1 = clock()
    print 'Time for optimization =',t1-t0
    print "Optimal nlml =", fopt

    ## PREDICTION
    vargout = gp(hyp2_opt,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2  = vargout[3]
    
    ## Plot results
    if PLOT:
        plotter(xstar,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])
    

    ##----------------------------------------------------------##
    ## SPARSE GP (example 4)                                    ##
    ##----------------------------------------------------------##   
    print '...example 4: FITC training and prediction...'
    ## SPECIFY inducing points
    n = x.shape[0]
    num_u = np.fix(n/2)
    u = np.linspace(-1.3,1.3,num_u).T
    u  = np.reshape(u,(num_u,1))

    ## SPECIFY FITC covariance function
    covfunc = [['kernels.covFITC'], covfunc, u]
    
    ## SPECIFY FICT inference method
    inffunc  = ['inf.infFITC']

    ## TRAINING: OPTIMIZE hyperparameters
    [hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2_opt,gp,'Minimize',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    #[hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2_opt,gp,'SCG',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    print 'Optimal F =', fopt

    ## FITC PREDICTION
    vargout = gp(hyp2_opt, inffunc, meanfunc, covfunc, likfunc, x, y, xstar)
    ymF = vargout[0]; y2F = vargout[1]; mF  = vargout[2];  s2F = vargout[3]
    
    ## Plot results
    if PLOT:
        FITCplotter(u,xstar,ymF,y2F,x,y,[-1.9, 1.9, -0.9, 3.9])
    
    
    
