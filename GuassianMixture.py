import numpy as np


class GMM:

    """

    """
    def __init__(self, k, eps):
        self.k = k
        self.eps = eps




    def em_step(self, X, iters):

        # m = rows, n = columns
        m, n = X.shape

        # mean / starting points
        mu = X[np.random.choice(n, self.k), :]

        # covariance matrix for each guassian
        sigma = [np.eye(m)] * self.k

        # weights for probabilities, if k = 2, then we have two probabilities
        w = [1./self.k] * self.k

        # responsibility matrix is initialized to all zeros
        # we have responsibility for each of n points for eack of k gaussians
        R = np.zeros((n, self.k))

        # log_likelihoods
        log_likelihoods = []

        P = lambda mu, s: np.linalg.det(s) ** -.5 ** (2 * np.pi) ** (-X.shape[1] / 2.) \
                          * np.exp(-.5 * np.einsum('ij, ij -> i',
                                                   X - mu, np.dot(np.linalg.inv(s), (X - mu).T).T))

        return P


np.random.seed(3)
m1, cov1 = [9, 8], [[.5, 1], [.25, 1]]  ## first gaussian
data1 = np.random.multivariate_normal(m1, cov1, 90)

m2, cov2 = [6, 13], [[.5, -.5], [-.5, .1]]  ## second gaussian
data2 = np.random.multivariate_normal(m2, cov2, 45)

m3, cov3 = [4, 7], [[0.25, 0.5], [-0.1, 0.5]]  ## third gaussian
data3 = np.random.multivariate_normal(m3, cov3, 65)
X = np.vstack((data1, np.vstack((data2, data3))))

m, n = X.shape


# mean / starting points
mu = X[np.random.choice(n, 3), :]

# covariance matrix for each guassian
sigma = [np.eye(m)] * 3

gmm = GMM(3, 0.00001)
print(gmm.em_step(X, iters=100))
