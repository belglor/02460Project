import numpy as np
import scipy as sp


def guassian(X, mu, s):
    # Maybe use einsum
    # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
    return 1.0 / (((2 * np.pi) ** (-X.shape[1] / 2.) * np.linalg.det(s)) ** 0.5) #        np.exp(-.5 * np.matrix(X - mu) * np.linalg.inv(s) * np.matrix(X - mu).T)


def em_gmm(X, k, eps, max_iters):

    m, n = X.shape

    mu = X[np.random.choice(n, k), :]

    sigma = [np.eye(m)] * k

    # initialize the probabilities/weights for each gaussians
    w = [1.0 / k] * k

    # responsibility matrix is initialized to all zeros
    # we have responsibility for each of n points for eack of k gaussians
    gamma = np.zeros((n, k))

    # log_likelihoods
    log_likelihoods = []

    P = guassian(X, mu, sigma)

    # Iterate till max_iters iterations
    while len(log_likelihoods) < max_iters:
        # Bishop 438-439 (PDF 455-456)
        # E Step

        # Vectorized implementation of e-step equation to calculate the
        # membership for each of k -gaussians
        for k in range(k):
            gamma[:, k] = w[k] * P(mu[k], sigma[k])

        # Likelihood computation
        log_likelihood = np.sum(np.log(np.sum(gamma, axis=1)))

        log_likelihoods.append(log_likelihood)

        # Normalize so that the responsibility matrix is row stochastic
        gamma = (gamma.T / np.sum(gamma, axis=1)).T

        # The number of datapoints belonging to each gaussian
        N_ks = np.sum(gamma, axis=0)

        # M Step
        # calculate the new mean and covariance for each gaussian by
        # utilizing the new responsibilities
        for k in range(k):

            # means
            mu[k] = 1.0 / N_ks[k] * np.sum(gamma[:, k] * X.T, axis=1).T
            x_mu = np.matrix(X - mu[k])

            # covariances
            sigma[k] = np.array(1.0 / N_ks[k] * np.dot(np.multiply(x_mu.T, gamma[:, k]), x_mu))

            # and finally the probabilities
            w[k] = 1.0 / n * N_ks[k]
        # check for onvergence
        if len(log_likelihoods) < 2:
            continue
        if np.abs(log_likelihood - log_likelihoods[-2]) < eps:
            break

    # bind all results together
    from collections import namedtuple
    params = namedtuple('params', ['mu', 'sigma', 'w', 'log_likelihoods', 'max_iters'])
    params.mu = mu
    params.sigma = sigma
    params.w = w
    params.log_likelihoods = log_likelihoods
    params.num_iters = len(log_likelihoods)

    return params


X = sp.append(sp.random.multivariate_normal([-3.5, 5.0], sp.eye(2)*4, 50), sp.random.multivariate_normal([-8.2, 10.0], sp.eye(2)*2, 70)).reshape(50+70, 2)

em_gmm(X, 3, 0.000001, 100)