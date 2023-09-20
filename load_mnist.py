import numpy as np


# Read binary MNIST files
def read_mnist_images(file_path):
    with open(file_path, "rb") as file:
        data = np.frombuffer(file.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28 * 28)


def read_mnist_labels(file_path):
    with open(file_path, "rb") as file:
        data = np.frombuffer(file.read(), dtype=np.uint8, offset=8)
    return data


# Read MNIST dataset files
train_images = read_mnist_images("mnist_data/train-images.idx3-ubyte")
train_labels = read_mnist_labels("mnist_data/train-labels.idx1-ubyte")
test_images = read_mnist_images("mnist_data/t10k-images.idx3-ubyte")
test_labels = read_mnist_labels("mnist_data/t10k-labels.idx1-ubyte")

# Confirm data is loaded
# print("Train Images Shape:", train_images.shape)  # Should be (num_samples, 28*28)
# print("Train Labels Shape:", train_labels.shape)  # Should be (num_samples,)
