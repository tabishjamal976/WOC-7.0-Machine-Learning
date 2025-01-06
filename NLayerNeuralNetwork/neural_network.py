import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', learning_rate=0.01, epochs=1000, batch_size=32, lambda_reg=0.01, dropout_rate=0.2, class_weights=None):
        """
        Initialize the neural network with given hyperparameters and layer sizes.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = np.float32(learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.lambda_reg = lambda_reg  
        self.dropout_rate = dropout_rate  
        self.class_weights = class_weights  

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size).astype('float32') * 0.1
        self.b1 = np.zeros((1, self.hidden_size), dtype='float32')
        self.W2 = np.random.randn(self.hidden_size, self.output_size).astype('float32') * 0.1
        self.b2 = np.zeros((1, self.output_size), dtype='float32')

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax activation function for multi-class classification."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X, training=True):
        """Perform a forward pass through the network."""
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1) if self.activation == 'relu' else self.sigmoid(self.Z1)

        # Apply dropout during training
        if training:
            self.dropout_mask = (np.random.rand(*self.A1.shape) > self.dropout_rate).astype('float32')
            self.A1 *= self.dropout_mask
        else:
            self.A1 *= (1 - self.dropout_rate)  # Scale during inference

        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2) if self.output_size > 1 else self.sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        """Perform a backward pass and update weights using gradient descent."""
        m = X.shape[0]

        # Compute gradients for output layer
        dz2 = self.A2 - y
        if self.class_weights is not None:
            dz2 *= self.class_weights
        dW2 = np.dot(self.A1.T, dz2) / m + self.lambda_reg * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Compute gradients for hidden layer
        dz1 = np.dot(dz2, self.W2.T) * (self.Z1 > 0)
        dW1 = np.dot(X.T, dz1) / m + self.lambda_reg * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def fit(self, X, y):
        """Train the model using gradient descent."""
        num_samples = X.shape[0]
        epsilon = 1e-7  # To prevent log(0)

        for epoch in range(self.epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, num_samples, self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            if epoch % 100 == 0:
                output_full = self.forward(X, training=False)
                loss = -np.mean(np.sum(y * np.log(output_full + epsilon), axis=1))
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """Predict class labels."""
        output = self.forward(X, training=False)
        return (output > 0.5).astype(int) if self.output_size == 1 else np.argmax(output, axis=1)

    def calculate_f1(self, actual, predicted):
        """Calculate the F1-score for multi-class classification."""
        if self.output_size == 1:
            actual = actual.flatten()
            predicted = predicted.flatten()
        else:
            actual = np.argmax(actual, axis=1)
        precision, recall, f1 = 0, 0, 0

        for cls in np.unique(actual):
            tp = np.sum((actual == cls) & (predicted == cls))
            fp = np.sum((actual != cls) & (predicted == cls))
            fn = np.sum((actual == cls) & (predicted != cls))

            p = tp / (tp + fp + 1e-7)
            r = tp / (tp + fn + 1e-7)
            f1 += 2 * (p * r) / (p + r + 1e-7)

        return f1 / len(np.unique(actual))
