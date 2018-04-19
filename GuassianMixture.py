from scipy.linalg import det, inv
import numpy as np


def gaussian_fun(x, u, sigma):
    x, u = np.array(x), np.array(u)
    x_mu = x-u
    return np.exp(-(np.dot(x_mu, np.dot(inv(sigma), x_mu)))/2.0) / (((2*np.pi)**(x.shape[0]/2.0)) * (det(sigma) ** 0.5))


def gmm(X, K, max_iter=10, eps=1e-6):
    """
    Gaussian Mixture Model With EM
    Arguments:
    - `X`: Input data (2D array).
    - `K`: Number of clusters.
    - `max_iter`: Number of iterations to run.
    - `eps`: Tolerance.
    """
    X = np.array(X)
    N, D = X.shape
    pi = np.ones(K) * 1.0/K
    pi = [1./K] * K
    mu = np.random.rand(K, D)
    sigma = np.array([np.eye(D) for i in range(K)])
    L = np.inf

    for i in range(max_iter):
        # E-step
        gamma = np.apply_along_axis(lambda x: np.fromiter((pi[k] * gaussian_fun(x, mu[k], sigma[k]) for k in range(K)), dtype=float), 1, X)
        gamma /= np.sum(gamma, 1)[:, np.newaxis]

        # M-step
        Nk = np.sum(gamma, 0)
        mu = np.sum(X*gamma.T[:, :, np.newaxis], 1) / Nk[Ellipsis, np.newaxis]
        x_mu = X[:, np.newaxis, :] - mu
        sigma = np.sum(gamma[Ellipsis, np.newaxis, np.newaxis] * x_mu[:, :, np.newaxis, :] * x_mu[:, :, :, np.newaxis], 0) / Nk[Ellipsis, np.newaxis, np.newaxis]
        pi = Nk / N

        # Likelihood
        Lnew = np.sum(np.log2(np.sum(np.apply_along_axis(lambda x: np.fromiter((pi[k] * gaussian_fun(x, mu[k], sigma[k]) for k in range(K)), dtype=float), 1, X), 1)))
        if abs(L-Lnew) < eps: break
        L = Lnew
        print("L=%s" % L)

    cls = np.zeros(N)
    for i in range(K):
        cls[gamma[:, i] > 1.0/K] = i

    return dict(pi=pi, mu=mu, sigma=sigma, gamma=gamma, classification=cls)


if __name__ == '__main__':
    X = np.append(np.random.multivariate_normal([-3.5, 5.0], np.eye(2)*4, 50),
                     np.random.multivariate_normal([-8.2, 10.0], np.eye(2)*2, 70)).reshape(50+70, 2)
    K = 2
    d = gmm(X, K)
    print("\npi = {}\n\nmu = {}\n\nsigma = {}\n\n".format(d['pi'], d['mu'], d['sigma']))
    gamma = d['gamma']
    # print( gamma)
    # from sklearn import mixture
    # g = mixture.GaussianMixture(n_components=2)
    # g_gmm = g.fit(X)
    # print("weights.{} og means{} og covar{}".format(g.weights_, g.means_, g.covariances_))