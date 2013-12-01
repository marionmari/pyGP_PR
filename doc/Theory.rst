Theory
======
Gaussian Processes (GPs) can conveniently be used for Bayesian supervised learning, such as regression and classification. 
In its simplest form, GP inference can be implemented in a few lines of code. However, in practice, things typically 
get a little more complicated: you might want to use complicated covariance functions and mean functions, learn good values 
for hyperparameters, use non-Gaussian likelihood functions (rendering exact inference intractable), use approximate inference 
algorithms, or combinations of many or all of the above. This is what the pyGP_FN software package does.

Before going straight to the examples, just a brief note about the organization of the package. There are four types of 
objects which you need to know about:

Gaussian Process
----------------
A Gaussian Process is fully specified by a mean function and a covariance function. These functions are specified separately, 
and consist of a specification of a functional form as well as a set of parameters called hyperparameters, see below.

Mean functions
^^^^^^^^^^^^^^
Several mean functions are available, all start with the four letters mean and reside in the means module. 

.. toctree::
   :maxdepth: 2

   meanAPI

Covariance functions
^^^^^^^^^^^^^^^^^^^^
There are many covariance functions available, all start with the three letters cov and reside in the kernels module. 

.. toctree::
   :maxdepth: 2

   kernelAPI

For both mean functions and covariance functions, two types exist: simple and composite. Whereas simple types are specified by the function 
name, composite functions join together several components using nested python lists. Composite functions can be composed of other composite 
functions, allowing for very flexible and interesting structures.

Hyperparameters
---------------
GPs are typically specified using mean and covariance functions which have free parameters called hyperparameters. Also 
likelihood functions may have such parameters. These are encoded in the :class:`src.Tools.utils.hyperParameters` class with the fields 
mean, cov and lik (some of which may be empty). When specifying hyperparameters, it is important that the number of elements in each of these 
fields precisely match the number of parameters expected by the mean function, the covariance function and the likelihood functions 
respectively (otherwise an error will result). Hyperparameters whose natural domain is positive are represented by their 
logarithms. 

Likelihood Functions
--------------------
The likelihood function specifies the probability of the observations given the latent function, i.e. the Gaussian Process (and the 
hyperparameters). All likelihood functions begin with the three letters lik and reside in the likelihoods module

.. toctree::
   :maxdepth: 2

   likelihoodAPI

Some examples are :func:`src.Core.likelihoods.likGauss`, the Gaussian likelihood for regression, and :func:`src.Core.likelihoods.likErf`, 
the Gaussian likelihood function for binary classification    

Inference Method
----------------
The inference methods specify how to compute with the model, i.e. how to infer the (approximate) posterior process, how to find 
hyperparameters, evaluate the log marginal likelihood and how to make predictions. Inference methods all begin with the three 
letters inf and reside in the inferences module. 

.. toctree::
   :maxdepth: 2

   inferenceAPI

Some examples are :func:`src.Core.inferences.infExact`, for exact inference (regression with Gaussian likelihood), 
:func:`src.Core.inferences.infFITC`, for large scale Gaussian regression, :func:`src.Core.inferences.infEP`, for the Expectation 
Propagation algorithm, :func:`src.Core.inferences.infVB` and :func:`src.Core.inferences.infKL` for two variational approximations 
based on a lower bound to the marginal likelihood, or :func:`src.Core.inferences.infFITC_Laplace` for large scale Laplace's approximation. 

