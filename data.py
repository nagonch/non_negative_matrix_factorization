import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def add_occlusion_block(image_array, b=10):
    block_index_i = np.random.randint(0, image_array.shape[0] - (b + 1))
    block_index_j = np.random.randint(0, image_array.shape[1] - (b + 1))
    result = np.copy(image_array)
    result[
        block_index_i : block_index_i + b, block_index_j : block_index_j + b
    ] = np.ones((b, b))
    return result


def add_salt_and_pepper_noise(image_array, ratio=0.025):
    for corrupted_value in [0, 1]:
        noise = np.random.uniform(
            low=0.0,
            high=1.0,
            size=(image_array.shape[0], image_array.shape[1]),
        )
        mask = (noise <= ratio).astype(np.float32)
        image_array = np.where(mask, corrupted_value, image_array)
    return image_array


def load_data(root="data/CroppedYaleB", reduce=1, corruption_type=None):
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.

    """
    images, labels, images_clean = [], [], []

    for i, person in enumerate(sorted(os.listdir(root))):
        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):
            # Remove background images in Extended YaleB dataset.
            if fname.endswith("Ambient.pgm"):
                continue

            if not fname.endswith(".pgm"):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert("L")  # grey image.

            # reduce computation complexity.
            img = img.resize([s // reduce for s in img.size])

            # convert image to numpy array.
            img = np.asarray(img)
            img = (img - img.min()) / (img.max() - img.min() + 1e-9)
            img_corrupted = img
            if corruption_type == "occlusion":
                img_corrupted = add_occlusion_block(img)
            elif corruption_type == "salt_and_pepper":
                img_corrupted = add_salt_and_pepper_noise(img)

            # collect data and label.
            images_clean.append(img)
            images.append(img_corrupted)
            labels.append(i)

    # concate all images and labels.
    images = np.array(images)
    images = images.reshape(-1, images.shape[1] * images.shape[2])
    images_clean = np.array(images_clean)
    images_clean = images_clean.reshape(
        -1, images_clean.shape[1] * images_clean.shape[2]
    )
    labels = np.array(labels)
    return (
        images.T,
        labels,
        images_clean.T,
    )


if __name__ == "__main__":
    images, labels = load_data(
        root="data/CroppedYaleB",
        corruption_type="occlusion",
    )

    print(images.shape, labels.shape)
