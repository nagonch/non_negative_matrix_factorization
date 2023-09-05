import os
import numpy as np
from PIL import Image
from numpy import sqrt
import matplotlib.pyplot as plt


def add_occlusion_block(image_array, b=10):
    block_index_i = np.random.randint(0, image_array.shape[0] - (b + 1))
    block_index_j = np.random.randint(0, image_array.shape[1] - (b + 1))
    result = np.copy(image_array)
    result[
        block_index_i : block_index_i + b, block_index_j : block_index_j + b
    ] = 255 * np.ones((b, b))
    return result


def add_salt_and_pepper_noise(image_array, ratio=0.025):
    for corrupted_value in [0, 255]:
        noise = np.random.uniform(
            low=0.0,
            high=1.0,
            size=(image_array.shape[0], image_array.shape[1]),
        )
        mask = (noise <= ratio).astype(np.float32)
        image_array = np.where(mask, corrupted_value, image_array)
    return image_array


def load_data(root="data/CroppedYaleB", reduce=4):
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.

    """
    images, labels = [], []

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
            img = add_occlusion_block(img)
            plt.imshow(img)
            plt.show()

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels


if __name__ == "__main__":
    load_data()
