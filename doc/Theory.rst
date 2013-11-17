Theory
======
Gaussian Processes (GPs) can conveniently be used for Bayesian supervised learning, such as regression and classification. 
In its simplest form, GP inference can be implemented in a few lines of code. However, in practice, things typically 
get a little more complicated: you might want to use complicated covariance functions and mean functions, learn good values 
for hyperparameters, use non-Gaussian likelihood functions (rendering exact inference intractable), use approximate inference 
algorithms, or combinations of many or all of the above. This is what the GPML software package does.

Before going straight to the examples, just a brief note about the organization of the package. There are four types of 
objects which you need to know about:

Gaussian Process
----------------
A Gaussian Process is fully specified by a mean function and a covariance function. These functions are specified separately, 
and consist of a specification of a functional form as well as a set of parameters called hyperparameters, see below.

Mean functions
^^^^^^^^^^^^^^
Several mean functions are available, all start with the four letters mean and reside in the mean directory. 

.. toctree::
   :maxdepth: 2

   meanAPI

Covariance functions
^^^^^^^^^^^^^^^^^^^^
There are many covariance functions available, all start with the three letters cov and reside in the cov directory. 
An overview is provided by the covFunctions help function (type help covFunctions to get help), and an example is the 
covSEard "Squared Exponential with Automatic Relevance Determination" covariance function. 

For both mean functions and covariance functions, two types exist: simple and composite. Whereas simple types are specified by the function name (or function pointer), composite functions join together several components using cell arrays. Composite functions can be composed of other composite functions, allowing for very flexible and interesting structures.
Examples are given below and in the usageMean and usageCov functions. 

Hyperparameters
---------------
GPs are typically specified using mean and covariance functions which have free parameters called hyperparameters. Also 
likelihood functions may have such parameters. These are encoded in a struct with the fields mean, cov and lik (some of which 
may be empty). When specifying hyperparameters, it is important that the number of elements in each of these struct fields, 
precisely match the number of parameters expected by the mean function, the covariance function and the likelihood functions 
respectively (otherwise an error will result). Hyperparameters whose natural domain is positive are represented by their 
logarithms. 

Likelihood Functions
--------------------
The likelihood function specifies the probability of the observations given the latent function, i.e. the GP (and the 
hyperparameters). All likelihood functions begin with the three letters lik and reside in the lik directory. An overview is 
provided by the likFunctions help function (type help likFunctions to get help). Some examples are likGauss the Gaussian 
likelihood for regression, likPoisson the Poisson likelihood for count data, likGamma the Gamma likelihood for positive data, 
likBeta the Beta likelihood for interval data, and likLogistic the logistic function used in classification (a.k.a. logistic 
regression). Output dependent noise can be modeled using warped GPs implemented by likGaussWarp. As for the mean and covariance 
functions there is also a composite likelihood likMix that can be used to generate mixtures of other likelihood functions.
    
Examples are given below and in the usageLik function. 

Inference Method
----------------
The inference methods specify how to compute with the model, i.e. how to infer the (approximate) posterior process, how to find 
hyperparameters, evaluate the log marginal likelihood and how to make predictions. Inference methods all begin with the three 
letters inf and reside in the inf directory. An overview is provided by the infMethods help file (type help infMethods to get 
help). Some examples are infExact for exact inference (regression with Gaussian likelihood), infFITC for large scale Gaussian 
regression, infEP for the Expectation Propagation algorithm, infVB and infKL for two variational approximations based on a lower 
bound to the marginal likelihood, or infFITC_Laplace for large scale Laplace's approximation. Further usage examples are provided 
for both regression and classification. However, not all combinations of likelihood function and inference method are possible 
(e.g. you cannot do exact inference with a Laplace likelihood). An exhaustive compatibility matrix between likelihoods (rows) 
and inference methods (columns) is given in the table below: 
