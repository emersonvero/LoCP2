from itertools import chain
import numpy as np
import pandas as pd
from scipy.stats import beta,invgamma,gamma,multivariate_normal,binom

def sampling(
        y, # y is shaped like (n_samples)
        X, # X is shaped like (n_samples,n_features)
        a1=0.01,
        a2=0.01,
        theta=0.5,
        a=1.,
        b=1.,
        s=0.5,
        chain_samples=6000,
        nr_burnin=1500
        ):

    n_samples = X.shape[0]
    n_features = X.shape[1]

    # dictionary of empty arrays to store different samples
    res = {
        "beta" : np.empty((chain_samples,n_features)),
        "z" : np.empty((chain_samples,n_features)),
        "sigma2" : np.empty(chain_samples),
        "tau2" : np.empty(chain_samples),
        "theta" : np.empty(chain_samples)
    }

    # initialize the masking as ones
    res["z"][0] = np.ones(n_features)
    # initialize the beta as least square regression
    res["beta"][0] = np.linalg.lstsq(X,y,rcond=None)[0]
    # initialize the sigma as the variance of the residuals
    res["sigma2"][0] = np.var(y - X @ res["beta"][0])
    # initialize the tau2 as one and the theta as 0.5
    res["tau2"][0] = 1.
    res["theta"][0] = 0.5

    # compute only once
    XtX = X.T @ X
    Xty = X.T @ y

    # ----------------- BEGIN SAMPLING

    for i in range(1,chain_samples):

        # lets retrieve the previous values for easier coding
        z_prev = res["z"][i-1]
        beta_prev = res["beta"][i-1]
        sigma2_prev = res["sigma2"][i-1]
        tau2_prev = res["tau2"][i-1]
        theta_prev = res["theta"][i-1]

        # ------------------ LETS GO WITH THE CONDITIONALS

        # sample theta from a Beta distribution
        theta_new = beta.rvs(a + np.sum(z_prev),b+np.sum(1-z_prev))

        # sample sigma2 from an inverse gamma
        err = y - X @ beta_prev
        scale = 1./(a2 + (err.T @ err)/2)
        sigma2_new = 1./gamma.rvs(a1+n_samples/2,scale=scale)

        # sample tau2 from an inverse gamma
        scale = 1./((s**2)/2 + (beta_prev.T @ beta_prev)/(2*sigma2_new))
        tau2_new = 1./gamma.rvs(0.5+0.5*np.sum(z_prev),scale=scale)

        # sample new beta from a multivariate gaussian
        covariance = np.linalg.inv(XtX/sigma2_new + np.eye(n_features)/(sigma2_new*tau2_new))
        mean = covariance @ Xty /sigma2_new # is this right?
        beta_new = multivariate_normal.rvs(mean = mean,cov=covariance)

        # now we sample the zjs
        # in random order
        for j in np.random.permutation(n_features):
            
            # grab the current vector
            z0 = z_prev
            # set j to zero
            z0[j] = 0.
            # get the beta_{-j}
            bz0 = beta_new * z0

            # compute the u variables (one for each sample)
            xj = X[:,j] # the jth feature of each sample
            u = y - X @ bz0 
            cond_var = np.sum(xj**2) + 1./tau2_new

            # compute the chance parameter:
            # the probability of extracting zj = 0 is prop to (1-theta)
            # while of extracting zj=1 is (.....) mess 
            # computing the logarithm of these (l0 and l1) means that the probability of extracting zj=1 is
            # xi = exp(l1)/(exp(l1)+exp(l0))
            # we can also write this as
            # xi = 1/(1+ exp(l0-l1))
            # this way we can check if exp(l0-l1) overflows and just call it xi = 0

            l0 = np.log(1-theta_new)
            l1 = np.log(theta_new) \
                - 0.5 * np.log(tau2_new*sigma2_new) \
                + (np.sum(xj*u)**2)/(2*sigma2_new*cond_var) \
                + 0.5*np.log(sigma2_new/cond_var)

            el0_l1 = np.exp(l0-l1)
            if np.isinf(el0_l1):
                xi = 0
            else:
                xi = 1/(1+el0_l1)
            
            # extract the zj
            z_prev[j]=binom.rvs(1,xi)

        # once we extracted all zj, store them:
        z_new = z_prev

        # update everything

        res["z"][i] = z_new
        res["beta"][i] = beta_new
        res["sigma2"][i] = sigma2_new
        res["tau2"][i] = tau2_new
        res["theta"][i] = theta_new

    # ---------- END SAMPLING

    for k in res.keys():
        res[k] = res[k][nr_burnin:]
    
    return res # dopo lo cambio per togliere il burnin



rng = np.random.RandomState(1234)
n_samples = 100

# X shaped like (n_samples,n_features)
X = rng.random(size=(n_samples,5))*10

# beta shaped like (n_features)
betas = np.array([
    1.,
    -2.,
    0.,
    4.,
    0.
    ])

sigma = 1

# y shaped like (n_samples)
y = X @ betas + rng.randn((n_samples))*sigma



samples = sampling(y,X)


import matplotlib.pyplot as plt

