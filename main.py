import load_mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the MNIST training data using load_mnist module
train_images = load_mnist.train_images
train_labels = load_mnist.train_labels

# Load the MNIST testing data using load_mnist module
test_images = load_mnist.test_images
test_labels = load_mnist.test_labels

# Create a KNN classifier
handwritten_digit_model = KNeighborsClassifier(n_neighbors=3)

# Train the model on the training data
# train_labels is the prediction target
# train_images is the features
handwritten_digit_model.fit(train_images, train_labels)

# Make predictions on the test data
predictions = handwritten_digit_model.predict(test_images)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Load and process a local image
image_path = "local_test_image/image_2.jpg"
img = Image.open(image_path).convert("L")  # Convert to grayscale
img = Image.fromarray(255 - np.array(img))  # Invert the colors
img = img.resize((28, 28))  # Resize to match MNIST image size
img = np.array(img).reshape(1, -1) / 255.0  # Normalize pixel values and flatten image

# Make prediction on local image
prediction = handwritten_digit_model.predict(img)
print(f"Classification of local image: {prediction[0]}")

# Display the local image and prediction
plt.imshow(img.reshape(28, 28), cmap=plt.cm.gray_r)
plt.title(f"Predicted Digit: {prediction[0]}")
plt.axis("off")
plt.show()
