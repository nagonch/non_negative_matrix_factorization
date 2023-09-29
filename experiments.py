from metrics import get_metrics
from data import load_data
from nmf import LeeSeungNMF
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


def run_experiment(dataset, method, noise_type):
    images, labels = load_data(
        root=f"data/{dataset}", corruption_type=noise_type
    )
    K = len(set(labels))
    nmf_method = method(K)
    D, R = nmf_method.fit(images)
    metrics = get_metrics(images, D, R, labels)

    return metrics


if __name__ == "__main__":
    algorithms = [LeeSeungNMF]
    noise_types = [None, "salt_and_pepper", "occlusion"]
    datasets = ["ORL", "CroppedYaleB"]
    data = []
    index = []
    for alg in algorithms:
        alg_name = alg.__name__
        for noise_type in noise_types:
            for dataset in datasets:
                print(f"Evaluating {alg_name}, {dataset}, {noise_type}...")
                metrics = run_experiment(dataset, alg, noise_type)
                index.append(
                    [
                        alg_name,
                        noise_type if noise_type is not None else "-",
                        dataset,
                    ]
                )
                data.append(
                    [
                        metrics.nmi,
                        metrics.rmse,
                        metrics.accuracy,
                    ]
                )
    index = pd.MultiIndex.from_tuples(
        index,
        names=[
            "algorithm",
            "noise_type",
            "dataset",
        ],
    )
    data = pd.DataFrame(
        data,
        columns=[
            "nmi",
            "rmse",
            "accuracy",
        ],
        index=index,
    )
    print("\n")
    print(data)
