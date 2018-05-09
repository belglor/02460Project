from scipy.linalg import det, inv
import numpy as np


def initialize(x, K):
    """

    :param x:
    :param K:
    :return:
    """
    x = np.asarray(x)
    n, d = x.shape
    pi = [1. / K] * K
    mu = np.random.rand(K, d)
    sigma = [np.eye(d)] * K
    L = np.inf
    return n, d, pi, mu, sigma, L


def _gaussian(x, mean, covariance):
    """

    :param x:
    :param mean:
    :param covariance:
    :return:
    """
    x, mu = np.asarray(x), np.asarray(mean)
    x_mu = x - mu
    return np.exp(-(np.dot(x_mu, np.dot(inv(covariance), x_mu)))) / (((2*np.pi) ** (x.shape[0]) * np.abs(det(covariance))) ** 0.5)

def expectation_maximization(X, K, max_iter=100, eps=1e-6):
    """

    :param X:
    :param K:
    :param max_iter:
    :param eps:
    :return:
    """
    n, d, pi, mu, sigma, L = initialize(X, K)


if __name__ == '__main__':
    X = np.append(np.random.multivariate_normal([-3.5, 5.0], np.eye(2)*4, 50),
                  np.random.multivariate_normal([-8.2, 10.0], np.eye(2)*2, 70)).reshape(50+70, 2)

    K = 2
    d = gmm(X, K)
    print("\npi = {}\n\nmu = {}\n\nsigma = {}\n\n".format(d['pi'], d['mu'], d['sigma']))
    gamma = d['gamma']
