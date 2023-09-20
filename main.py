import load_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
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
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate mean absolute error
mae = mean_absolute_error(test_labels, predictions)
print(f"Mean Absolute Error: {mae:.2f}")
