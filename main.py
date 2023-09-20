import load_mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Load and process a local image
    image_path = "local_test_image/image_7.jpg"
    img = load_and_process_image(image_path)

    # Load the MNIST training data
    train_images = load_mnist.train_images
    train_labels = load_mnist.train_labels

    # Create a KNN classifier
    handwritten_digit_model = KNeighborsClassifier(n_neighbors=3)

    # Train the model on the training data
    handwritten_digit_model.fit(train_images, train_labels)

    # Predict handwritten digit in local image
    prediction = handwritten_digit_model.predict(img)

    # Get the indices of the nearest neighbors
    nearest_neighbors_indices = handwritten_digit_model.kneighbors(
        img, n_neighbors=3, return_distance=False
    )

    # Retrieve the nearest neighbor images
    nearest_neighbor_images = train_images[nearest_neighbors_indices[0]]

    # Display the local image and its nearest neighbors
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(img.reshape(28, 28), cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.title(f"Local Image\nPredicted Digit: {prediction[0]}")
    plt.axis("off")

    for i, neighbor_index in enumerate(nearest_neighbors_indices[0]):
        plt.subplot(1, 4, i + 2)
        plt.imshow(
            nearest_neighbor_images[i].reshape(28, 28),
            cmap=plt.cm.gray,
            vmin=0,
            vmax=255,
        )
        plt.title(f"Neighbor {i + 1}\nTrue Digit: {train_labels[neighbor_index]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def load_and_process_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = Image.fromarray(255 - np.array(img))  # Invert the colors
    img = img.resize((28, 28))  # Resize to match MNIST image size
    img = np.array(img).reshape(1, -1)

    # Normalize local image colors
    return scale_image_colors(img)


def scale_image_colors(img: np.ndarray) -> np.ndarray:
    """
    Scale the colors of an image to be between 0 and 255 for accurate model prediction.
    """
    numpy_min = np.min(img)
    numpy_max = np.max(img)
    numpy_diff = numpy_max - numpy_min
    img = (img - numpy_min) * (255 / numpy_diff)
    return img


if __name__ == "__main__":
    main()
