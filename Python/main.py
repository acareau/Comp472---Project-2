import numpy as np

class NeuralNetwork:
    def __init__(self, inputs, hidden_size, outputs):
        self.inputs = inputs
        self.hidden_size = hidden_size
        self.outputs = outputs

        # randomly initialize weights and biases for the hidden and output layers
        self.hidden_weights = np.random.randn(self.inputs, self.hidden_size)
        self.hidden_bias = np.zeros((1, self.hidden_size))
        self.weights_output = np.random.randn(self.hidden_size, self.outputs)
        self.bias_output = np.zeros((1, self.outputs))

    def forward(self, X):
        # Forward propagation through the network

        # Hidden layer activation
        self.hidden_layer_activation = np.dot(X, self.hidden_weights) + self.hidden_bias
        self.hidden_layer_output = self.relu(self.hidden_layer_activation)

        # Output layer activation
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_activation)

        return self.output_layer_output

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def mean_squared_error(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # forward pass
            output = self.forward(X)

            # get the loss
            loss = self.mean_squared_error(output, y)

            # backpropagation below!

            # calculate difference between output and actual answer
            d_output = output - y

            # Calculate the derivative of the loss with respect to the hidden layer
            d_hidden = np.dot(d_output, self.weights_output.T) * (self.hidden_layer_output > 0)

            # Update weights and biases using gradient descent
            self.weights_output -= learning_rate * np.dot(self.hidden_layer_output.T, d_output)
            self.bias_output -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
            self.hidden_weights -= learning_rate * np.dot(X.T, d_hidden)
            self.hidden_bias -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

            # track progress through loss
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")


# training X
X = np.array([[1, 1, 1, 0, 0],
    [1,1,1,0,1],
    [0,0,0,1,0],
    [0,0,0,1,1],
    [0,0,1,0,0],
    [0,0,1,0,1],
    [0,0,1,1,0],
    [0,0,1,1,1],
    [0,1,0,0,0],
    [0,1,0,0,1],
    [0,1,0,1,0],
    [0,1,0,1,1],
    [0,1,1,0,0],
    [0,1,1,0,1],
    [0,1,1,1,0],
    [0,1,1,1,1],
    [1,0,0,0,0],
    [1,0,0,0,1],
    [1,0,0,1,0],
    [1,0,0,1,1],
    [1,0,1,0,0],
    [1,0,1,0,1],
    [1,0,1,1,0],
    [1,0,1,1,1]])
# training answers
y = np.array([[1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1]])


# create the network with input size 5, hidden layers 2 and output size 1
nn = NeuralNetwork(inputs=5, hidden_size=5, outputs=1)

# train the network using our X
nn.train(X, y, epochs=100, learning_rate=0.1)

# test the new inputs
new_input = np.array([[1,1,0,0,0],
    [1,1,0,0,1],
    [0,0,0,0,0],
    [0,0,0,0,1],
    [1,1,0,1,0],
    [1,1,0,1,1],
    [1,1,1,1,0],
    [1,1,1,1,1]])
prediction = nn.forward(new_input)

print("Predictions: ")
print(prediction)
