import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_info_score,
    mean_squared_error,
)
import numpy.typing as npt


@dataclass
class NMFMetrics:
    rmse: float
    accuracy: float
    nmi: float


def assign_cluster_label(W: npt.NDArray[np.float32], Y: npt.NDArray[np.uint8]):
    Y_pred = np.argmax(W, axis=1)
    for i in set(Y_pred):
        ind = Y_pred == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def get_metrics(
    X_original: npt.NDArray[np.uint8],
    W: npt.NDArray[np.float32],
    H: npt.NDArray[np.float32],
    S: npt.NDArray[np.float32],
    Y: npt.NDArray[np.uint8],
):
    X_reconstructed = W @ H + S
    rmse = np.sqrt(np.mean((X_original - X_reconstructed) ** 2))
    cluster_labels = assign_cluster_label(W, Y)
    acc = accuracy_score(Y, cluster_labels)
    nmi = normalized_mutual_info_score(Y, cluster_labels)

    return NMFMetrics(rmse, acc, nmi)


if __name__ == "__main__":
    pass
