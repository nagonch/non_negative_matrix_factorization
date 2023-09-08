import numpy as np
import numpy.typing as npt
from data import load_data
from tqdm import tqdm


class LeeSeungNMF:
    def __init__(
        self,
        latent_space_size: int,
        n_iterations: int = 500,
    ):
        self.K = latent_space_size
        self.n_iterations = n_iterations

    def preprocess(self, X, result_mean=0.25, result_std=0.25):
        self.N, self.M = X.shape
        self.W = np.random.uniform(size=(self.N, self.K))
        self.H = np.random.uniform(size=(self.K, self.M))
        current_mean = X.mean()
        current_std = X.std()
        scaled_matrix = (X - current_mean) * (
            result_std / current_std
        ) + result_mean
        return np.clip(scaled_matrix, 0, 1)

    def fit(
        self,
        X: npt.NDArray[np.uint8],
    ):
        X = self.preprocess(X)
        for n in tqdm(range(self.n_iterations)):
            A = self.W.T @ X
            B = self.W.T @ self.W @ self.H
            for i in range(self.H.shape[0]):
                for j in range(self.H.shape[1]):
                    self.H[i][j] *= A[i][j] / B[i][j]
            C = X @ self.H.T
            D = self.W @ self.H @ self.H.T
            for i in range(self.W.shape[0]):
                for j in range(self.W.shape[1]):
                    self.W[i][j] *= C[i][j] / D[i][j]
        return self.W, self.H


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    images, labels = load_data(root="data/ORL", corruption_type=None)
    K = 2
    alg = LeeSeungNMF(K)
    W, H = alg.fit(images)
    for i in range(K):
        plt.imshow(H[i, :].reshape(28, 23))
        plt.show()
