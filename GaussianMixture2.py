from scipy.linalg import det, inv
import numpy as np


def gaussian(X, mu, sigma):
    """

    :param X:
    :param mu:
    :param sigma:
    :retmurn:
    """
    X, mu = np.array(X), np.array(mu)
    x_mu = X - mu
    return np.exp(-(np.dot(x_mu, np.dot(inv(sigma), x_mu)))/2.0) / (((2*np.pi) ** (X.shape[0] / 2.0)) * (det(sigma) ** 0.5))


def initialize(X, K):
    """

    :param X:
    :param K:
    :return:
    """
    X = np.array(X)
    N, D = X.shape
    pi = [1. / K] * K
    mu = np.random.rand(K, D)
    sigma = [np.eye(D)] * K
    L = np.inf

    return N, D, pi, mu, sigma, L


def e_step(X, K, mu, sigma, pi):
    """

    :param X:
    :param K:
    :param mu:
    :param sigma:
    :param pi:
    :return:
    """
    gamma_inner = np.apply_along_axis(lambda X: np.fromiter((pi[k] * gaussian(X, mu[k], sigma[k]) for k in range(K)), dtype=float), 1, X)
    gamma_inner /= np.sum(gamma_inner, 1)[:, np.newaxis]
    return gamma_inner


def m_step(X, N, gamma):
    """

    :param X:
    :param N:
    :param gamma:
    :return:
    """
    Nk = np.sum(gamma, 0)
    mu = np.sum(X * gamma.T[:, :, np.newaxis], 1) / Nk[Ellipsis, np.newaxis]
    x_mu = X[:, np.newaxis, :] - mu
    sigma = np.sum(gamma[Ellipsis, np.newaxis, np.newaxis] * x_mu[:, :, np.newaxis, :] * x_mu[:, :, :, np.newaxis], 0) \
            / Nk[Ellipsis, np.newaxis, np.newaxis]
    pi = Nk / N

    return mu, sigma, pi


def EM(X, K, max_iter=10, eps=1e-6):
    """

    :param X:
    :param K:
    :param max_iter:
    :param eps:
    :return:
    """

    N, D, pi, mu, sigma, lk = initialize(X, K)

    for i in range(max_iter):
        # E-step
        gamma = e_step(X, K, mu, sigma, pi)

        # M-step
        mu, sigma, pi = m_step(X, N, gamma)

        # Likelihood
        lk_new = np.sum(np.log2(np.sum(np.apply_along_axis(lambda x: np.fromiter((pi[k] * gaussian(x, mu[k], sigma[k]) for k in range(K)), dtype=float), 1, X), 1)))

        if np.abs(lk-lk_new) < eps:
            break
        lk = lk_new
        # print("LogLikelihood = {}".format(lk))

    cls = np.zeros(N)
    for i in range(K):
        cls[gamma[:, i] > 1.0/K] = i

    return dict(pi=pi, mu=mu, sigma=sigma, gamma=gamma, lk=lk, classification=cls)


if __name__ == '__main__':
    X = np.append(np.random.multivariate_normal([-3.5, 5.0], np.eye(2)*4, 50),
                     np.random.multivariate_normal([-8.2, 10.0], np.eye(2)*2, 70)).reshape(50+70, 2)
    K = 2
    d = EM(X, K)
    print("\npi = {}\n\nmu = {}\n\nsigma = {}\n\nloglikelihood = {}\n\n".format(d['pi'], d['mu'], d['sigma'], d['lk']))
    # gamma = d['gamma']
