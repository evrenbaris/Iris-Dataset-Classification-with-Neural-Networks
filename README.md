# Iris Dataset Classification with Neural Networks

This project demonstrates the classification of the Iris dataset using an Artificial Neural Network (ANN) implemented with TensorFlow. The project includes data preprocessing, model training, evaluation, and data visualization in both 2D and 3D using PCA.

## Features
- Implementation of an Artificial Neural Network (ANN) for classification.
- Visualization of the dataset in 2D and 3D using Principal Component Analysis (PCA).
- High accuracy in classifying the Iris dataset.

## Tools and Libraries Used
- `numpy`
- `matplotlib`
- `tensorflow`
- `scikit-learn`

## How to Run
1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install numpy matplotlib tensorflow scikit-learn
   ```
3. Run the script:
   ```bash
   python iris_classification.py
   ```

## Steps in the Project
1. **Data Preprocessing**:
   - Load the Iris dataset using `scikit-learn`.
   - Scale the features for better performance.
   - Split the dataset into training and testing sets.
2. **Model Creation and Training**:
   - Build an ANN model with one hidden layer and a softmax output layer.
   - Train the model using categorical cross-entropy loss and the Adam optimizer.
3. **Evaluation**:
   - Evaluate the model on the test dataset to compute accuracy.
4. **Visualization**:
   - Use PCA to reduce the dataset dimensions to 2D and 3D.
   - Visualize real and predicted classes in 2D and 3D plots.



