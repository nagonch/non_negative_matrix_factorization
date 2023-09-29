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
        verbose: bool = True,
    ):
        self.K = latent_space_size
        self.n_iterations = n_iterations
        self.verbose = verbose

    def preprocess(
        self,
        X: npt.NDArray[np.uint8],
        result_mean: float = 0.25,
        result_std: float = 0.25,
    ):
        self.N, self.M = X.shape
        self.W = np.random.uniform(size=(self.N, self.K))
        self.H = np.random.uniform(size=(self.K, self.M))
        current_mean = X.mean()
        current_std = X.std()
        scaled_matrix = (X - current_mean) * (
            result_std / current_std
        ) + result_mean  # descibed in "Methods" section of the paper
        return np.clip(scaled_matrix, 0, 1)

    def fit(
        self,
        X: npt.NDArray[np.uint8],
        eps: float = 1e-6,
    ):
        X = self.preprocess(X)
        iterator = range(self.n_iterations)
        if self.verbose:
            iterator = tqdm(iterator)
        for _ in iterator:
            self.H *= (self.W.T @ X) / (self.W.T @ self.W @ self.H + eps)
            self.W *= (X @ self.H.T) / (self.W @ self.H @ self.H.T + eps)
        return self.W, self.H


class RobustNMF:
    """
    Zhang, Lijun, et al. "Robust non-negative matrix factorization."
    Frontiers of Electrical and Electronic Engineering in China 6 (2011): 192-200.
    """

    def __init__(
        self,
        latent_space_size: int,
        n_iterations: int = 500,
        lambda_reg: float = 0.7,
        verbose: bool = True,
    ):
        self.K = latent_space_size
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.verbose = verbose

    def preproces(
        self,
        X: npt.NDArray[np.uint8],
        result_mean: float = 0.25,
        result_std: float = 0.25,
    ):
        self.N, self.M = X.shape
        self.W = np.random.uniform(size=(self.N, self.K))
        self.H = np.random.uniform(size=(self.K, self.M))
        current_mean = X.mean()
        current_std = X.std()
        scaled_matrix = (X - current_mean) * (
            result_std / current_std
        ) + result_mean  # descibed in "Methods" section of the paper
        return np.clip(scaled_matrix, 0, 1)

    def fit(self, X: npt.NDArray[np.uint8], eps: float = 1e-6):
        X = self.preproces(X)
        iterator = range(self.n_iterations)
        if self.verbose:
            iterator = tqdm(iterator)
        for _ in iterator:
            S = X - self.W @ self.H
            S = np.where(
                S > self.lambda_reg / 2,
                S - self.lambda_reg / 2,
                np.where(S < -self.lambda_reg / 2, S + self.lambda_reg / 2, 0),
            )
            self.W = (
                (np.abs((S - X) @ self.H.T) - ((S - X) @ self.H.T))
                / (2 * (self.W @ self.H @ self.H.T) + eps)
            ) * self.W

            self.H = (
                (np.abs(self.W.T @ (S - X)) - self.W.T @ (S - X))
                / (2 * self.W.T @ self.W @ self.H + eps)
            ) * self.H
            normalization = (
                np.sqrt(np.sum(self.W**2, axis=0, keepdims=True)) + eps
            )
            self.W /= normalization
            self.H *= normalization.T

        return self.W, self.H


if __name__ == "__main__":
    images, labels = load_data(root="data/ORL", corruption_type=None)
    K = 20
    alg = RobustNMF(K, lambda_reg=0.7)
    W, H = alg.fit(images)
    for i in range(K):
        plt.imshow(
            H[i, :].reshape(28, 23),
            # H[i, :].reshape(48, 42),
            cmap="gray",
            interpolation="bicubic",
            aspect="auto",
        )
        plt.show()
