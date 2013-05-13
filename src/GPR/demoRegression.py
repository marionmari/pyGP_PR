

import numpy as np
import matplotlib.pyplot as plt
from gp import gp
from Tools.min_wrapper import min_wrapper 



from UTIL.utils import convert_to_array, hyperParameters, plotter, FITCplotter

if __name__ == '__main__':
    
    # TODO    
    ### GENERATE data from a noisy GP and GENERATE sample observations from the GP
    #n = 20      # number of labeled/training data
    #D = 1       # dimension of input data


    ### DATA
    x = np.array([[2.083970427750732,  -0.821018066101379,  -0.617870699182597,  -1.183822608860694,\
                  0.274087442277144,   0.599441729295593,   1.768897919204435,  -0.465645549031928,\
                  0.588852784375935,  -0.832982214438054,  -0.512106527960363,   0.277883144210116,\
                  -0.065870426922211,  -0.821412363806325,   0.185399443778088,  -0.858296174995998,\
                   0.370786630037059,  -1.409869162416639,-0.144668412325022,-0.553299615220374]]).T
    
    y = np.array([[4.549203746331698,   0.371985574437271,   0.711307965514790,  -0.013212893618430,   2.255473255338191,\
                  1.009915749295733,   3.744675937965029,   0.424592771793202,   1.322833652295811,   0.278298293510020,\
                  0.267229130945574,   2.200112286723833,   1.200609983308969,   0.439971697236094,   2.628580433511255,\
                  0.503774817336353,   1.942525313820564,   0.579133950013327,   0.670874423968554,   0.377353755100965]]).T


    #### PLOT data
    #plt.plot(x,y,'b+',markersize=12)
    #plt.axis([-1.9,1.9,-0.9,3.9])
    #plt.grid()
    #plt.xlabel('input x')
    #plt.ylabel('output y')
    #plt.show()

    ### TEST points
    xstar = np.array([np.linspace(-2,2,101)]).T             # test points evenly distributed in the interval [-2, 2]
    
    ### DEFINE parameterized mean and covariance functions
    covfunc  = [ ['kernels.covPoly'] ]
    meanfunc = [ ['means.meanSum'], [ ['means.meanLinear'] , ['means.meanConst'] ] ]
    ### DEFINE likelihood function used
    inffunc  = ['inf.infExact']
    ### SPECIFY inference method
    likfunc  = ['lik.likGauss']

    ## SET (hyper)parameters
    hyp = hyperParameters()
    hyp.cov = np.array([np.log(0.25),np.log(1.0),3.0])
    hyp.mean = np.array([0.5,1.0])
    hyp.lik = np.array([np.log(0.1)])

    ###----------------------------------------------------------###
    ### STANDARD GP (example 1)                                  ###
    ###----------------------------------------------------------###
    ### PREDICTION 
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2 = vargout[3]
    
    #### PLOT results
    #plotter(xstar,ym,ys2,x,y,[-2, 2, -0.9, 3.9])
    
    ### GET negative log marginal likelihood
    [nlml, post] = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)
    print "nlml = ", nlml[0]
    
    
    ###----------------------------------------------------------###
    ### STANDARD GP (example 2)                                  ###
    ###----------------------------------------------------------###
    ### USE another covariance function
    covfunc = [ ['kernels.covSEiso'] ]
    
    ### SET (hyper)parameters
    hyp2 = hyperParameters()
    hyp2.cov = np.array([-1.0,0.0])
    hyp2.mean = np.array([0.5,1.0])
    hyp2.lik = np.array([np.log(0.1)])

    ### PREDICTION 
    vargout = gp(hyp2,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2 = vargout[3]
    
    #### PLOT results
    #plotter(xstar,ym,ys2,x,y,[-2, 2, -0.9, 3.9])
    
    ### GET negative log marginal likelihood
    [nlml, post] = gp(hyp2,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)
    print "nlml2 = ", nlml[0]


    ###----------------------------------------------------------###
    ### STANDARD GP (example 3)                                  ###
    ###----------------------------------------------------------###
    ### TRAINING: OPTIMIZE hyperparameters
    [hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2,gp,'CG',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    print "nlml_opt = ", fopt
    
    #[hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2,gp,'BFGS',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    #print fopt

    ### PREDICTION
    vargout = gp(hyp2_opt,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2  = vargout[3]
    
    ### Plot results
    plotter(xstar,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])
    

    ###----------------------------------------------------------###
    ### SPARSE GP (example 4)                                    ###
    ###----------------------------------------------------------###    
    ### SPECIFY FITC covariance function
    n = x.shape[0]
    num_u = np.fix(n/2)
    u = np.linspace(-1.3,1.3,num_u).T
    u  = np.reshape(u,(num_u,1))
    covfuncF = [['kernels.covFITC'], covfunc, u]
    
    ### SPECIFY FICT inference method
    inffuncF  = ['inf.infFITC']
    
    ### FICT PREDICTION
    vargout = gp(hyp2_opt, inffuncF, meanfunc, covfuncF, likfunc, x, y, xstar);
    ymF = vargout[0]; y2F = vargout[1]; mF  = vargout[2];  s2F = vargout[3]
    
    ### Plot results
    FITCplotter(u,xstar,ymF,y2F,x,y,[-1.9, 1.9, -0.9, 3.9])
    
    
    
