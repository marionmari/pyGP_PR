Classification 
------------------------------------
This example recreates the classification example from the `GPML`_ package.

The difference between regression and classification is not of a fundamental nature. We can still use a Gaussian process latent function 
in essentially the same way, but unfortunately, the Gaussian likelihood function often used for regression is inappropriate for classification. 
Since exact inference is only possible for Gaussian likelihood, we need an alternative, approximate, inference method.

Here, we will demonstrate binary classification, using two partially overlapping Gaussian sources of data in two dimensions. 
First we import the data (so that it exactly matches the example in GPML)::

    ## LOAD data
    demoData = np.load('../../data/classification_data.npz')
    x = demoData['x']            # training data
    y = demoData['y']            # training target
    xstar = demoData['xstar']    # test data
    n = xstar.shape[0]           # number of test points

:math:`120` data points are generated from two Gaussians with different means and covariances. One Gaussian is isotropic and contains 
:math:`2/3` of the data (blue), the other is highly correlated and contains :math:`1/3` of the points (red). 
Note, that the labels for the targets are :math:`\pm 1` (and not :math:`0/1`).

In the plot, we superimpose the data points with the posterior equi-probability contour lines for the probability of the second class
given complete information about the generating mechanism.

.. figure:: images/demoC1.png
   :align: center

We specify a Gaussian process model as follows: a constant mean function, :func:`src.Core.means.meanConst` with initial parameter set to 
:math:`0`, a squared exponential 
with automatic relevance determination (ARD) covariance function :func:`src.Core.kernels.covSEard`. This covariance function has one 
characteristic length-scale parameter for each dimension of the input space (:math:`2` total), and a signal magnitude parameter, for 
a total of :math:`3` hyperparameters. ARD with separate length-scales for each input dimension is a very powerful tool to learn which 
inputs are important for predictions: if length-scales are short, inputs are very important, and when they grow very long 
(compared to the spread of the data), the corresponding inputs will be largely ignored. Both length-scales and the signal magnitude 
are initialized to :math:`1` (and represented in the log space). Finally, the likelihood function :func:`src.Core.likelihoods.likErf` 
has the shape of the error-function (or cumulative Gaussian), which doesn't take any hyperparameters (so hyp.lik does not exist)::

    ## DEFINE parameterized mean and covariance functions
    meanfunc = ['means.meanConst']
    covfunc  = ['kernels.covSEard']
    ## DEFINE likelihood function used 
    likfunc = ['likelihoods.likErf']
    ## SPECIFY inference method
    inffunc = ['inferences.infLaplace']

    ## SET (hyper)parameters
    hyp = hyperParameters()
    hyp.mean = np.array([0.])
    hyp.cov  = np.array([0.,0.,0.])
    [hyp_opt, fopt, gopt, funcCalls] = min_wrapper(hyp,gp,'Minimize',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)

    hyp = hyp_opt

We train the hyperparameters using minimize, to minimize the negative log marginal likelihood. We allow for :math:`100` function evaluations, 
and specify that inference should be done with the Expectation Propagation (EP) inference method :func:`src.Core.inferences.infEP`, and pass 
the usual parameters. 
Training is done using algorithm 3.5 and 5.2 from the `GPML`_ book. When computing test probabilities, we call gp with additional test inputs, 
and as the last argument a vector of targets for which the log probabilities lp should be computed. The fist four output arguments 
of the function are mean and variance for the targets and corresponding latent variables respectively. The test set predictions are 
computed using algorithm 3.6 from the GPML book. The contour plot for the predictive distribution is shown below. Note, that the predictive 
probability is fairly close to the probabilities of the generating process in regions of high data density. Note also, that as you move 
away from the data, the probability approaches :math:`1/3`, the overall class probability.

.. figure:: images/demoC2.png
   :align: center

Examining the two ARD characteristic length-scale parameters after learning, you will find that they are fairly similar, reflecting the fact 
that for this data set, both inputs are important.

Large scale classification
--------------------------
In case the number of training inputs :math:`x` exceeds a few hundred, approximate inference using :func:`src.Core.inferences.infLaplace`, 
:func:`src.Core.inferences.infEP` and :func:`src.Core.inferences.infVB` takes too long. As in regression, we offer the FITC approximation 
based on a low-rank plus diagonal approximation to the exact covariance to deal with these cases. The general idea is to use inducing points 
:math:`u` and to base the computations on cross-covariances between training, test and inducing points only.

Using the FITC approximation is very simple, we just have to wrap the covariance function covfunc into :func:`src.Core.kernels.covFITC` 
and call :func:`src.Core.gp` with the inference methods :func:`src.Core.inferences.infFITC_Laplace` and :func:`src.Core.inferences.infFITC_EP` 
as demonstrated by the following lines of code::

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
    [hyp_opt, fopt, gopt, funcCalls] = min_wrapper(hyp,gp,'Minimize',inffunc,meanfunc,covfuncF,likfunc,x,y,None,None,True)  # minimize by Carl$
    print 'Optimal nlml =', fopt

    ## FITC PREDICTION     
    [ymu,ys2,fmu,fs2,lp,post] = gp(hyp_opt, inffunc, meanfunc, covfuncF, likfunc, x, y, xstar, np.ones((n,1)) )

We define equispaced inducing points :math:`u` that are shown in the figure as black circles. Alternatively, a random subset of the training 
points can be used as inducing points.

.. figure:: images/demoC3.png
   :align: center

.. _GPML: http://www.gaussianprocess.org/gpml/code/matlab/doc/

