Regression Demonstration Example
--------------------------------
This example recreates the regression example from the `GPML <http://www.gaussianprocess.org/gpml/code/matlab/doc/>`_ package.


This is a simple example, where we first generate n=20 data points from a Gaussian Process (GP), where the inputs are scalar (so that it is easy to plot what is going on). We then use various other GPs to make inferences about the underlying function.

First, generate the exact data from the GPML example (this data is hardcoded in the /data folder).

Above, we first specify the mean function meanfunc, covariance function covfunc of a GP and a likelihood function, likfunc. The corresponding hyperparameters are specified in the hyp structure:

The mean function is composite, adding (using meanSum function) a linear (meanLinear) and a constant (meanConst) to get an affine function. Note, how the different components are composed using cell arrays. The hyperparameters for the mean are given in hyp.mean and consists of a single (because the input will one dimensional, i.e. D=1) slope (set to 0.5) and an off-set (set to 1). The number and the order of these hyperparameters conform to the mean function specification. You can find out how many hyperparameters a mean (or covariance or likelihood function) expects by calling it without arguments, such as feval(@meanfunc{:}). For more information on mean functions see meanFunctions and the directory mean/.

The covariance function is of the Matérn form with isotropic distance measure covMaterniso. This covariance function is also composite, as it takes a constant (related to the smoothness of the GP), which in this case is set to 3. The covariance function takes two hyperparameters, a characteristic length-scale ell and the standard deviation of the signal sf. Note, that these positive parameters are represented in hyp.cov using their logarithms. For more information on covariance functions see covFunctions and cov/.

Finally, the likelihood function is specified to be Gaussian. The standard deviation of the noise sn is set to 0.1. Again, the representation in the hyp.lik is given in terms of its logarithm. For more information about likelihood functions, see likFunctions and lik/.

Then, we generate a dataset with n=20 examples. The inputs x are drawn from a unit Gaussian (using the gpml_randn utility, which generates unit Gaussian pseudo random numbers with a specified seed). We then evaluate the covariance matrix K and the mean vector m by calling the corresponding functions with the hyperparameters and the input locations x. Finally, the targets y are computed by drawing randomly from a Gaussian with the desired covariance and mean and adding Gaussian noise with standard deviation exp(hyp.lik). The above code is a bit special because we explicitly call the mean and covariance functions (in order to generate samples from a GP); ordinarily, we would only directly call the gp function.
f1.gif

Let's ask the model to compute the (joint) negative log probability (density) nlml (also called marginal likelihood or evidence) and to generalize from the training data to other (test) inputs z:

  nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y)

  z = linspace(-1.9, 1.9, 101)';
  [m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

  f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
  fill([z; flipdim(z,1)], f, [7 7 7]/8)
  hold on; plot(z, m); plot(x, y, '+')

The gp function is called with a struct of hyperparameters hyp, and inference method, in this case @infExact for exact inference and the mean, covariance and likelihood functions, as well as the inputs and outputs of the training data. With no test inputs, gp returns the negative log probability of the training data, in this example nlml=11.97.

To compute the predictions at test locations we add the test inputs z as a final argument, and gp returns the mean m variance s2 at the test location. The program is using algorithm 2.1 from the GPML book. Plotting the mean function plus/minus two standard deviations (corresponding to a 95% confidence interval):
f2.gif

Typically, we would not a priori know the values of the hyperparameters hyp, let alone the form of the mean, covariance or likelihood functions. So, let's pretend we didn't know any of this. We assume a particular structure and learn suitable hyperparameters:

  covfunc = @covSEiso; hyp2.cov = [0; 0]; hyp2.lik = log(0.1);

  hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfunc, likfunc, x, y);
  exp(hyp2.lik)
  nlml2 = gp(hyp2, @infExact, [], covfunc, likfunc, x, y)

  [m s2] = gp(hyp2, @infExact, [], covfunc, likfunc, x, y, z);
  f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
  fill([z; flipdim(z,1)], f, [7 7 7]/8)
  hold on; plot(z, m); plot(x, y, '+')

First, we guess that a squared exponential covariance function covSEiso may be suitable. This covariance function takes two hyperparameters: a characteristic length-scale and a signal standard deviation (magnitude). These hyperparameters are non-negative and represented by their logarithms; thus, initializing hyp2.cov to zero, correspond to unit characteristic length-scale and unit signal standard deviation. The likelihood hyperparameter in hyp2.lik is also initialized. We assume that the mean function is zero, so we simply ignore it (and when in the following we call gp, we give an empty argument for the mean function).

In the following line, we optimize over the hyperparameters, by minimizing the negative log marginal likelihood w.r.t. the hyperparameters. The third parameter in the call to minimize limits the number of function evaluations to a maximum of 100. The inferred noise standard deviation is exp(hyp2.lik)=0.15, somewhat larger than the one used to generate the data (0.1). The final negative log marginal likelihood is nlml2=14.13, showing that the joint probability (density) of the training data is about exp(14.13-11.97)=8.7 times smaller than for the setup actually generating the data. Finally, we plot the predictive distribution.
f3.gif

This plot shows clearly, that the model is indeed quite different from the generating process. This is due to the different specifications of both the mean and covariance functions. Below we'll try to do a better job, by allowing more flexibility in the specification.

Note that the confidence interval in this plot is the confidence for the distribution of the (noisy) data. If instead you want the confidence region for the underlying function, you should use the 3rd and 4th output arguments from gp as these refer to the latent process, rather than the data points.

  hyp.cov = [0; 0]; hyp.mean = [0; 0]; hyp.lik = log(0.1);
  hyp = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
  [m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);
 
  f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
  fill([z; flipdim(z,1)], f, [7 7 7]/8)
  hold on; plot(z, m); plot(x, y, '+');

Here, we have changed the specification by adding the affine mean function. All the hyperparameters are learnt by optimizing the marginal likelihood.
f4.gif

This shows that a much better fit is achieved when allowing a mean function (although the covariance function is still different from that of the generating process).
Large scale regression

In case the number of training inputs x exceeds a few thousands, exact inference using infExact.m takes too long. We offer the FITC approximation based on a low-rank plus diagonal approximation to the exact covariance to deal with these cases. The general idea is to use inducing points u and to base the computations on cross-covariances between training, test and inducing points only.

Using the FITC approximation is very simple, we just have to wrap the covariance function covfunc into covFITC.m and call gp.m with the inference method infFITC.m as demonstrated by the following lines of code.

  nu = fix(n/2); u = linspace(-1.3,1.3,nu)';
  covfuncF = {@covFITC, {covfunc}, u};
  [mF s2F] = gp(hyp, @infFITC, meanfunc, covfuncF, likfunc, x, y, z);

We define equispaced inducing points u that are shown in the figure as black circles. Note that the predictive variance is overestimated outside the support of the inducing inputs. In a multivariate example where densely sampled inducing inputs are infeasible, one can simply use a random subset of the training points.

  nu = fix(n/2); iu = randperm(n); iu = iu(1:nu); u = x(iu,:);

f5.gif

Exercises for the reader

Inference Methods
    Try using Expectation Propagation instead of exact inference in the above, by exchanging @infExact with @infEP. You get exactly identical results, why? 
Mean or Covariance
    Try training a GP where the affine part of the function is captured by the covariance function instead of the mean function. That is, use a GP with no explicit mean function, but further additive contributions to the covariance. How would you expect the marginal likelihood to compare to the previous case? 
