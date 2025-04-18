import argparse
import os
from nmf import LeeSeungNMF, RobustL1NMF, RobustNMF
from sklearn.cluster import KMeans
from data import load_data
from matplotlib import pyplot as plt


def get_nmf_results(images, labels, destination_folder, alg, downsample):
    shape = (
        (112 // downsample, 92 // downsample)
        if "ORL" in destination_folder
        else (192 // downsample, 168 // downsample)
    )
    K = len(set(labels))
    alg = RobustNMF(K)
    W, H, E = alg.fit(images)
    for i in range(K):
        plt.imshow(
            W.T[i, :].reshape(*shape),
            cmap="gray",
            interpolation="bicubic",
            aspect="auto",
        )
        plt.savefig(f"{destination_folder}/{str(i).zfill(4)}")


def get_results(noise_type, alg_name, dataset, algorithm, downsample):
    if alg_name != "KMEANS":
        destination_folder = f"experiments/{dataset}_{noise_type}_{alg_name}"
        if not os.path.exists(destination_folder):
            os.mkdir(destination_folder)
        images, labels, images_clean = load_data(
            root=f"data/{dataset}",
            corruption_type=noise_type,
            reduce=downsample,
        )
        get_nmf_results(
            images, labels, destination_folder, algorithm, downsample
        )


def all_experiments(downsample):
    if not os.path.exists("experiments"):
        os.mkdir("experiments")
    algorithms = [LeeSeungNMF, RobustNMF, RobustL1NMF, "KMEANS"]
    noise_types = [None, "salt_and_pepper", "occlusion"]
    datasets = ["ORL", "CroppedYaleB"]
    for alg in algorithms:
        alg_name = alg if type(alg) is str else alg.__name__
        for noise_type in noise_types:
            for dataset in datasets:
                print(
                    f"Evaluating {alg_name}, {dataset}, {noise_type if not noise_type is None else 'no noise'}..."
                )
                get_results(noise_type, alg_name, dataset, alg, downsample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downsample", type=int, default=2)

    args = parser.parse_args()
    all_experiments(args.downsample)
