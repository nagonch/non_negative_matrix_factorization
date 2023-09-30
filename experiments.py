from metrics import get_metrics
from data import load_data
from nmf import LeeSeungNMF, RobustNMF, RobustL1NMF
import warnings
import pandas as pd
import numpy as np
import argparse
import numpy.typing as npt

warnings.filterwarnings("ignore")


def run_experiment(dataset: npt.NDArray[np.uint8], method, noise_type: str):
    images, labels = load_data(root=f"data/{dataset}", corruption_type=noise_type)
    K = len(set(labels))
    nmf_method = method(K)
    W, H, S = nmf_method.fit(images)
    metrics = get_metrics(images, W, H, S, labels)

    return metrics


def all_experiments():
    algorithms = [LeeSeungNMF, RobustNMF, RobustL1NMF]
    noise_types = [None, "salt_and_pepper", "occlusion"]
    datasets = [
        "ORL",
    ]  # "CroppedYaleB"]
    data = []
    index = []
    for alg in algorithms:
        alg_name = alg.__name__
        for noise_type in noise_types:
            for dataset in datasets:
                print(
                    f"Evaluating {alg_name}, {dataset}, {noise_type if not noise_type is None else 'no noise'}..."
                )
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


def main(save_results: bool = False):
    data, index = all_experiments()
    get_results(data, index, save_csv=save_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-csv", action="store_true")

    args = parser.parse_args()
    main(args.save_csv)
