import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode the output labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the ANN model
model = Sequential([
    Dense(8, activation='relu', input_dim=4),  # Hidden layer
    Dense(3, activation='softmax')  # Output layer (3 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=1)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict class labels
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

# Visualize the real classes using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
for i, label in enumerate(data.target_names):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], label=label)

plt.title("Iris Dataset Visualization (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# Visualize the predicted classes
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=predicted_classes, cmap='viridis', label="Predictions", alpha=0.6)
plt.title("Predicted Classes (Iris Dataset)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Classes")
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10, label='Predictions')])
plt.show()

# Visualize in 3D using PCA
pca_3d = PCA(n_components=3)
X_reduced_3d = pca_3d.fit_transform(X)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, label in enumerate(data.target_names):
    ax.scatter(X_reduced_3d[y == i, 0], X_reduced_3d[y == i, 1], X_reduced_3d[y == i, 2], label=label)

ax.set_title("Iris Dataset Visualization (3D PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
ax.legend()
plt.show()
