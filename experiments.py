from metrics import get_metrics, get_k_means_metrics, NMFMetrics
from data import load_data
from nmf import LeeSeungNMF, RobustNMF, RobustL1NMF
import warnings
import pandas as pd
import numpy as np
import argparse
import numpy.typing as npt

warnings.filterwarnings("ignore")


def run_experiment(
    dataset: npt.NDArray[np.uint8],
    method,
    noise_type: str,
    num_iterations=5,
    random_fraction=0.85,
):
    images_data, labels_data, images_clean = load_data(
        root=f"data/{dataset}", corruption_type=noise_type
    )
    n_items = int(labels_data.shape[0] * random_fraction)
    inds = np.arange(0, labels_data.shape[0], 1)
    metrics_array = []
    for _ in range(num_iterations):
        np.random.shuffle(inds)
        images = images_data[:, inds][:, :n_items]
        clean_images = images_clean[:, inds][:, :n_items]
        labels = labels_data[inds][:n_items]
        if method == "KMEANS":
            metrics = get_k_means_metrics(images, labels)
        else:
            K = len(set(labels))
            nmf_method = method(K)
            W, H, S = nmf_method.fit(images)
            metrics = get_metrics(clean_images, W, H, S, labels)
        metrics_array.append(metrics)

    result_metrics = NMFMetrics(0, 0, 0)
    result_metrics.rmse = np.mean([metric.rmse for metric in metrics_array])
    result_metrics.nmi = np.mean([metric.nmi for metric in metrics_array])
    result_metrics.accuracy = np.mean(
        [metric.accuracy for metric in metrics_array]
    )

    return result_metrics


def all_experiments(
    iterations_per_eval: int = 5,
    sampling_fraction: float = 0.85,
):
    algorithms = [LeeSeungNMF, RobustNMF, RobustL1NMF, "KMEANS"]
    noise_types = [None, "salt_and_pepper", "occlusion"]
    datasets = ["CroppedYaleB", "CroppedYaleB"]
    data = []
    index = []
    for alg in algorithms:
        alg_name = alg if type(alg) is str else alg.__name__
        for noise_type in noise_types:
            for dataset in datasets:
                print(
                    f"Evaluating {alg_name}, {dataset}, {noise_type if not noise_type is None else 'no noise'}..."
                )
                metrics = run_experiment(
                    dataset,
                    alg,
                    noise_type,
                    iterations_per_eval,
                    sampling_fraction,
                )
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
    return data, index


def get_results(
    data: npt.NDArray[np.uint8],
    index: npt.NDArray[np.uint8],
    save_csv: bool = False,
):
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
    if save_csv:
        data.reset_index().to_csv("results.csv")
    print("\n")
    print(data)


def main(
    save_results: bool = False,
    iterations_per_eval: int = 5,
    sampling_fraction: float = 0.85,
):
    data, index = all_experiments(iterations_per_eval, sampling_fraction)
    get_results(data, index, save_csv=save_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--iterations-per-eval", type=int, default=5)
    parser.add_argument("--sampling-fraction", type=float, default=0.85)

    args = parser.parse_args()
    main(args.save_csv, args.iterations_per_eval, args.sampling_fraction)
