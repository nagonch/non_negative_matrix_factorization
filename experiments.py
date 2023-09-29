from metrics import get_metrics
from data import load_data
from nmf import LeeSeungNMF
import warnings

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
    for alg in algorithms:
        for noise_type in noise_types:
            for dataset in datasets:
                print(noise_type, dataset)
                print(run_experiment(dataset, alg, noise_type))
