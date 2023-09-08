import numpy as np
import numpy.typing as npt
from data import load_data
from tqdm import tqdm
from matplotlib import pyplot as plt


class LeeSeungNMF:
    """
    Lee, D. D., & Seung, H. S. (1999)
    Learning the parts of objects by non-negative matrix factorization
    Nature, 401(6755), 788-791
    """

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
        for _ in tqdm(range(self.n_iterations)):
            self.H *= (self.W.T @ X) / (self.W.T @ self.W @ self.H)
            self.W *= (X @ self.H.T) / (self.W @ self.H @ self.H.T)
        return self.W, self.H


if __name__ == "__main__":
    images, labels = load_data(root="data/ORL", corruption_type=None)
    K = 2
    alg = LeeSeungNMF(K)
    W, H = alg.fit(images)
    for i in range(K):
        plt.imshow(H[i, :].reshape(28, 23))
        plt.show()
