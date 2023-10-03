import numpy as np
import numpy.typing as npt
from data import load_data
from tqdm import tqdm
from matplotlib import pyplot as plt


class NMFBase:
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
    ):
        self.N, self.M = X.shape
        self.W = np.random.uniform(size=(self.N, self.K))
        self.H = np.random.uniform(size=(self.K, self.M))
        return X

    def fit(
        self,
        X: npt.NDArray[np.uint8],
        eps: float = 1e-6,
    ):
        raise NotImplementedError()


class LeeSeungNMF(NMFBase):
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
        super().__init__(latent_space_size, n_iterations, verbose)

    def fit(
        self,
        X: npt.NDArray[np.uint8],
        eps: float = 1e-9,
    ):
        X = self.preprocess(X)
        iterator = range(self.n_iterations)
        if self.verbose:
            iterator = tqdm(iterator)
        for _ in iterator:
            self.H *= (self.W.T @ X) / (self.W.T @ self.W @ self.H + eps)
            self.W *= (X @ self.H.T) / (self.W @ self.H @ self.H.T + eps)
        return self.W, self.H, np.zeros_like(X)


class RobustNMF(NMFBase):
    """
    Zhang, Lijun, et al. "Robust non-negative matrix factorization."
    Frontiers of Electrical and Electronic Engineering in China 6 (2011): 192-200.
    """

    def __init__(
        self,
        latent_space_size: int,
        n_iterations: int = 500,
        verbose: bool = True,
        lambda_reg: float = 0.7,
    ):
        super().__init__(latent_space_size, n_iterations, verbose)
        self.lambda_reg = lambda_reg

    def fit(self, X: npt.NDArray[np.uint8], eps: float = 1e-9):
        X = self.preprocess(X)
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

        return self.W, self.H, S


class RobustL1NMF(NMFBase):
    def __init__(
        self,
        latent_space_size: int,
        n_iterations: int = 500,
        verbose: bool = True,
        l1_ratio: float = 1.0,
        alpha: float = 0.1,
    ):
        super().__init__(latent_space_size, n_iterations, verbose)
        self.l1_ratio = l1_ratio
        self.alpha = alpha

    def fit(self, X: npt.NDArray[np.uint8], eps: float = 1e-9):
        X = self.preprocess(X)
        self.E = np.zeros_like(X)
        iterator = range(self.n_iterations)
        if self.verbose:
            iterator = tqdm(iterator)
        for _ in iterator:
            # Update H
            WH = np.dot(self.W, self.H)
            self.H *= (np.dot(self.W.T, X - self.E)) / (
                np.dot(self.W.T, WH) + eps
            )

            # Update W
            WH = np.dot(self.W, self.H)
            self.W *= (np.dot(X - self.E, self.H.T)) / (
                np.dot(WH, self.H.T) + eps
            )

            # Update E
            WH = np.dot(self.W, self.H)
            self.E = X - WH
            self.E = np.sign(self.E) * np.maximum(
                np.abs(self.E) - self.alpha * self.l1_ratio, 0
            )

        return self.W, self.H, self.E


if __name__ == "__main__":
    # pass
    images, labels = load_data(root="data/ORL", corruption_type=None)
    K = 20
    alg = RobustNMF(K)
    W, H, E = alg.fit(images)
    for i in range(K):
        plt.imshow(
            W.T[i, :].reshape(28, 23),
            # W.T[i, :].reshape(48, 42),
            cmap="gray",
            interpolation="bicubic",
            aspect="auto",
        )
        plt.show()
