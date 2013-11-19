Regression Demonstration Example
--------------------------------
This example exactly recreates the regression example from the `GPML <http://www.gaussianprocess.org/gpml/code/matlab/doc/>`_ package.

This is a simple example, where :math:`n=20` data points are from a Gaussian Process (GP). The inputs 
are scalar (so plotting is easy). We then use various other GPs to make inferences about the underlying function.

First, generate the exact data from the GPML example (this data is hardcoded in data/regression_data.npz).

First specify the mean function meanfunc, covariance function covfunc of a GP and a likelihood function, likfunc. The 
corresponding hyperparameters are specified in the :class:`src.Tools.utils.hyperParameters` class.

The mean function is composite, adding (using :func:`src.Core.means.meanSum` function) a linear (:func:`src.Core.means.meanLinear`) and a 
constant (:func:`src.Core.means.meanConst`) to get an affine function. Note, how the different components are composed using python
lists.
The hyperparameters for the mean are given in hyp.mean and consists of a single (because the input will one dimensional, i.e. 
:math:`D=1`) slope (set to :math:`0.5`) and an off-set (set to :math:`1`). The number and the order of these hyperparameters conform to 
the mean function specification. You can find out how many hyperparameters a mean (or covariance or likelihood function) expects by 
calling it without arguments, such as ``feval(meanfunc)``. 

For more information on mean functions see meanFunctions_.

The covariance function is of the Matern form with isotropic distance measure :func:`src.Core.kernels.covMatern`. This covariance 
function is also composite, as it takes a constant (related to the smoothness of the GP), which in this case is set to :math:`3`. The covariance function takes 
two hyperparameters, a characteristic length-scale :math:`L` and the standard deviation of the signal :math:`\sigma_f`. Note, that these positive 
parameters are represented in hyp.cov using their logarithms. For more information on covariance functions see covFunctions_.

Finally, the likelihood function is specified to be Gaussian. The standard deviation of the noise :math:`\sigma_n` is set to :math:`0.1`. Again, 
the representation in the hyp.lik is given in terms of its logarithm. For more information about likelihood functions, see 
likFunctions_.

Then, we import the dataset with :math:`n=20` examples. The inputs :math:`x` were drawn from a unit Gaussian. We then evaluate the 
covariance matrix :math:`K` and the mean vector :math:`m` by calling the corresponding functions with the hyperparameters and the input locations :math:`x`. Finally, 
the targets :math:`y` are imported, 
and they were drawn randomly from a Gaussian with the desired covariance and mean and adding Gaussian noise with standard deviation 
:math:`\exp(hyp.lik)`::

    ## LOAD DATA
    demoData = np.load('../../data/regression_data.npz')
    x = demoData['x']            # training data
    y = demoData['y']            # training target
    xstar = demoData['xstar']    # test data

The above code is a bit special because we explicitly load the data from a file (in order to 
generate the same samples as were generated in GPML); ordinarily, we would only directly call the :func:`src.Core.gp` function.

.. figure:: images/demoR1.png
   :align: center

Let's ask the model to compute the (joint) negative log probability (density) nlml (also called marginal likelihood, or evidence) and to 
generalize from the training data to other (test) inputs :math:`xstar`::

    ## PREDICTION
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2 = vargout[3]

    ## GET negative log marginal likelihood
    [nlml, post] = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)

The :func:`src.Core.gp` function is called with an instance of the hyperparameters class, hyp, and inference method.  In this example 
:func:`src.Core.inferences.infExact` for exact inference.  The  mean, covariance and likelihood functions, as well as the inputs and outputs of the training data are also used as arguments. With no test inputs, 
:func:`src.Core.gp` returns the negative log probability of the training data, in this example nlml:math:`=11.97`.

To compute the predictions at test locations, we add the test inputs :math:`z` as a final argument, and :func:`src.Core.gp` returns the mean :math:`m`,
and the variance :math:`\sigma^2` at the test location(s). Plotting the mean function plus/minus two standard 
deviations (corresponding to a :math:`95%` confidence interval):

.. figure:: images/demoR2.png
   :align: center

Typically, we would not *a priori* know the values of the hyperparameters hyp, let alone the form of the mean, covariance or likelihood 
functions. So, let's pretend we didn't know any of this. We assume a particular structure and learn suitable hyperparameters::

    covfunc = [ ['kernels.covSEiso'] ]

    ### SET (hyper)parameters
    hyp2 = hyperParameters()
    hyp2.cov = np.array([-1.0,0.0])
    hyp2.mean = np.array([0.5,1.0])
    hyp2.lik = np.array([np.log(0.1)])

    ## PREDICTION
    vargout = gp(hyp2,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2 = vargout[3]

    ## GET negative log marginal likelihood
    [nlml, post] = gp(hyp2,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)

First, we guess that a squared exponential covariance function :func:`src.Core.kernels.covSEiso` may be suitable. This covariance 
function takes two hyperparameters: a characteristic length-scale and a signal standard deviation (magnitude). These hyperparameters are 
non-negative and represented by their logarithms; thus, initializing hyp2.cov to zero, correspond to unit characteristic length-scale 
and unit signal standard deviation. The likelihood hyperparameter in hyp2.lik is also initialized. We assume that the mean function is 
zero, so we simply ignore it (and when in the following we call :func:`src.Core.gp`, we give an empty list for the mean function).

In the following line, we optimize over the hyperparameters, by minimizing the negative log marginal likelihood w.r.t. the 
hyperparameters. The third parameter in the call to :func:`src.Tools.minimize` limits the number of function evaluations (to a maximum of 
:math:`100`). The 
inferred noise standard deviation is :math:`\exp(hyp2.lik)=0.15`, somewhat larger than the one used to generate the data :math:`(0.1)`. The final 
negative log marginal likelihood is nlml2:math:`=14.13`, showing that the joint probability (density) of the training data is about 
:math:`\exp(14.13-11.97)=8.7` times smaller than for the setup actually generating the data. Finally, we plot the predictive distribution.

.. figure:: images/demoR3.png
   :align: center

This plot shows clearly, that the model is indeed quite different from the generating process. This is due to the different 
specifications of both the mean and covariance functions. Below we'll try to do a better job, by allowing more flexibility in 
the specification.

Note that the confidence interval in this plot is the confidence for the distribution of the (noisy) data. If instead you want 
the confidence region for the underlying function, you should use the 3rd and 4th output arguments from :func:`src.Core.gp` as these refer to the 
latent process, rather than the data points::

    ## TRAINING: OPTIMIZE HYPERPARAMETERS      
    ## -> parameter training via off-the-shelf optimization   
    [hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2,gp,'Minimize',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True) 

    ## PREDICTION
    vargout = gp(hyp2_opt,inffunc,meanfunc,covfunc,likfunc,x,y,xstar)
    ym = vargout[0]; ys2 = vargout[1]; m  = vargout[2]; s2  = vargout[3]

Here, we have changed the specification by adding the affine mean function. All the hyperparameters are learnt by optimizing 
the marginal likelihood.

.. figure:: images/demoR4.png
   :align: center

This shows that a much better fit is achieved when allowing a mean function (although the covariance function is still different 
from that of the generating process).

Large scale regression
----------------------
In case the number of training inputs :math:`x` exceeds a few thousands, exact inference using :func:`src.Core.inferences.infExact` takes 
too long. 
Instead, the FITC approximation based on a low-rank plus diagonal approximation to the exact covariance is used to deal with these cases. The general idea is 
to use inducing points :math:`u` and to base the computations on cross-covariances between training, test and inducing points only.

Using the FITC approximation is very simple, we just have to wrap the covariance function covfunc into :func:`src.Core.kernels.covFITC` 
and call :func:`src.Core.gp` with the inference method :func:`src.Core.inferences.infFITC` as demonstrated by the following lines of 
code::

    ## SPECIFY inducing points
    n = x.shape[0]
    num_u = np.fix(n/2)
    u = np.linspace(-1.3,1.3,num_u).T
    u  = np.reshape(u,(num_u,1))

    ## SPECIFY FITC covariance function
    covfunc = [['kernels.covFITC'], covfunc, u]

    ## SPECIFY FICT inference method
    inffunc  = ['inferences.infFITC']

    ## TRAINING: OPTIMIZE hyperparameters
    [hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2_opt,gp,'Minimize',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    #[hyp2_opt, fopt, gopt, funcCalls] = min_wrapper(hyp2_opt,gp,'SCG',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    print 'Optimal F =', fopt

    ## FITC PREDICTION
    vargout = gp(hyp2_opt, inffunc, meanfunc, covfunc, likfunc, x, y, xstar)
    ymF = vargout[0]; y2F = vargout[1]; mF  = vargout[2];  s2F = vargout[3]


We define equi-spaced inducing points :math:`u` that are shown in the figure as black X's. Note that the predictive variance is 
overestimated outside the support of the inducing inputs. In a multivariate example where densely sampled inducing inputs 
are infeasible, one can simply use a random subset of the training points.

.. figure:: images/demoR5.png
   :align: center


