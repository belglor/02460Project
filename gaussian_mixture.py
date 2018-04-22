from scipy.linalg import det, inv
import numpy as np




if __name__ == '__main__':
    X = np.append(np.random.multivariate_normal([-3.5, 5.0], np.eye(2)*4, 50),
                     np.random.multivariate_normal([-8.2, 10.0], np.eye(2)*2, 70)).reshape(50+70, 2)
    predict_proba(X)