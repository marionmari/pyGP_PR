Classification Demonstration Example
------------------------------------
This example recreates the classification example from the `GPML <http://www.gaussianprocess.org/gpml/code/matlab/doc/>`_ package.

You can either follow the example here on this page, or use the script demoClassification.

The difference between regression and classification isn't of fundamental nature. We can use a Gaussian process latent function in essentially the same way, it is just that the Gaussian likelihood function often used for regression is inappropriate for classification. And since exact inference is only possible for Gaussian likelihood, we also need an alternative, approximate, inference method.

Here, we will demonstrate binary classification, using two partially overlapping Gaussian sources of data in two dimensions. First we generate the data:

  clear all, close all
 
  n1 = 80; n2 = 40;                   % number of data points from each class
  S1 = eye(2); S2 = [1 0.95; 0.95 1];           % the two covariance matrices
  m1 = [0.75; 0]; m2 = [-0.75; 0];                            % the two means
 
  x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);
  x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2);
 
  x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';
  plot(x1(1,:), x1(2,:), 'b+'); hold on;
  plot(x2(1,:), x2(2,:), 'r+');

120 data points are generated from two Gaussians with different means and covariances. One Gaussian is isotropic and contains 2/3 of the data (blue), the other is highly correlated and contains 1/3 of the points (red). Note, that the labels for the targets are ±1 (and not 0/1).

In the plot, we superimpose the data points with the posterior equi-probability contour lines for the probability of class two given complete information about the generating mechanism

  [t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
  t = [t1(:) t2(:)]; n = length(t);                 % these are the test inputs
  tmm = bsxfun(@minus, t, m1');
  p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));
  tmm = bsxfun(@minus, t, m2');
  p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));
  contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9]);

f6.gif

We specify a Gaussian process model as follows: a constant mean function, with initial parameter set to 0, a squared exponential with automatic relevance determination (ARD) covariance function covSEard. This covariance function has one characteristic length-scale parameter for each dimension of the input space, and a signal magnitude parameter, for a total of 3 parameters (as the input dimension is D=2). ARD with separate length-scales for each input dimension is a very powerful tool to learn which inputs are important for predictions: if length-scales are short, inputs are very important, and when they grow very long (compared to the spread of the data), the corresponding inputs will be largely ignored. Both length-scales and the signal magnitude are initialized to 1 (and represented in the log space). Finally, the likelihood function likErf has the shape of the error-function (or cumulative Gaussian), which doesn't take any hyperparameters (so hyp.lik does not exist).

  meanfunc = @meanConst; hyp.mean = 0;
  covfunc = @covSEard; ell = 1.0; sf = 1.0; hyp.cov = log([ell ell sf]);
  likfunc = @likErf;

  hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
  [a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));

  plot(x1(1,:), x1(2,:), 'b+'); hold on; plot(x2(1,:), x2(2,:), 'r+')
  contour(t1, t2, reshape(exp(lp), size(t1)), [0.1:0.1:0.9]);

We train the hyperparameters using minimize, to minimize the negative log marginal likelihood. We allow for 40 function evaluations, and specify that inference should be done with the Expectation Propagation (EP) inference method @infEP, and pass the usual parameters. Training is done using algorithm 3.5 and 5.2 from the gpml book. When computing test probabilities, we call gp with additional test inputs, and as the last argument a vector of targets for which the log probabilities lp should be computed. The fist four output arguments of the function are mean and variance for the targets and corresponding latent variables respectively. The test set predictions are computed using algorithm 3.6 from the GPML book. The contour plot for the predictive distribution is shown below. Note, that the predictive probability is fairly close to the probabilities of the generating process in regions of high data density. Note also, that as you move away from the data, the probability approaches 1/3, the overall class probability.
f7.gif

Examining the two ARD characteristic length-scale parameters after learning, you will find that they are fairly similar, reflecting the fact that for this data set, both inputs important.
Large scale classification

In case the number of training inputs x exceeds a few hundreds, approximate inference using infLaplace.m, infEP.m and infVB.m takes too long. As in regression, we offer the FITC approximation based on a low-rank plus diagonal approximation to the exact covariance to deal with these cases. The general idea is to use inducing points u and to base the computations on cross-covariances between training, test and inducing points only.

Using the FITC approximation is very simple, we just have to wrap the covariance function covfunc into covFITC.m and call gp.m with the inference methods infFITC_Laplace.m and infFITC_EP.m as demonstrated by the following lines of code.

  [u1,u2] = meshgrid(linspace(-2,2,5)); u = [u1(:),u2(:)]; clear u1; clear u2
  nu = size(u,1);
  covfuncF = {@covFITC, {covfunc}, u};
  inffunc = @infFITC_EP;                       % also @infFITC_Laplace is possible
  hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfuncF, likfunc, x, y);
  [a b c d lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, t, ones(n,1));

We define equispaced inducing points u that are shown in the figure as black circles. Alternatively, a random subset of the training points can be used as inducing points.
f8.gif

Exercise for the reader

Inference Methods
    Use the Laplace Approximation for inference @infLaplace (or @infFITC_Laplace in the large scale example), and compare the approximate marginal likelihood for the two methods. Which approximation is best? 
Covariance Function
    Try using the squared exponential with isotropic distance measure covSEiso instead of ARD distance measure covSEard. Which is best? 
