#### Alessandro notes:

- in sampling, we initialize z to be all ones. maybe we should do all zeros? <- doesnt matter

  actually i dont think it matters, the overwhelming quantity of data points makes the posterior disregard completely such hyperparameters and initializations.

- amazingly, by adding noise to the data the algorithm works better, but still the masking vectors are all 1. 
  how can we mke the algorithm more prone to setting some variables to zero instead of just sampling near zero from a gaussian?

- maybe letting the model learn the variance of the slab prior allows it to sample it small enough to make it kind of a spiky gaussian, that is more probable because it accounts for small variations?


- PUSH THETA PRIOR TO BE CLOSE TO 0 -> sparsity

- HELP THE SAMPLER NOT GET STUCK IN WELLS AND HELP HIM FIND THE SPIKES ---> either by different initialization 

OBSERVATIONS:

differentiation method matters -> it may introduce some bias in the feature space

but WHY would it introduce some bias on xy? isnt it supposed to only compute the time derivative?
maybe it just so happens that the error on xdot correlates with xy so that the bias is introduced 

ROSSLER SYSTEM WITH ps.FiniteDifference(order=4) as differentiator works almost perfectly

prior over theta is basically inferring the MODEL SIZE:
i.e. if E[theta] < 0.5, we expect that less than half of the predictors are important

-> obtaining the best result is an interplay between setting the right prior for theta
(pushing the distribution towards zero, because we expect very few features to be active)
and reducing the bias introduced by the differentiation method 
(the posterior distribution is very sensible to such biases because of the term xizi in calculating the posterior distirbution of z)



1) choose a PDE and generate data

2) build "library", i.e. the features that supposedly build up our unknown PDE
this includes some choice of representation

3) regression magic

4) compare results


NOTES:

time dependent parameter to tune the system to make it go between a lorenz and a rossler

bayesian inference -> sparsity assumption has limits -> confidence levels

https://pysindy.readthedocs.io/en/latest/examples/13_ensembling.html

prior -> library construction and sparsity assumption
