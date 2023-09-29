import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score,
    normalized_mutual_info_score,
    mean_squared_error,
)


@dataclass
class NMFMetrics:
    rmse: float
    accuracy: float
    nmi: float


def assign_cluster_label(X, Y):
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def get_metrics(X_original, D, R, Y):
    X_reconstructed = D @ R
    rmse = mean_squared_error(
        X_original.reshape(-1), X_reconstructed.reshape(-1), squared=False
    )
    cluster_labels = assign_cluster_label(D, Y)
    acc = accuracy_score(Y, cluster_labels)
    nmi = normalized_mutual_info_score(Y, cluster_labels)

    return NMFMetrics(rmse, acc, nmi)


if __name__ == "__main__":
    pass
