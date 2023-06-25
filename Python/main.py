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

        # initialize output layer and hidden layer output/activation
        self.output_layer_output = None
        self.output_layer_activation = None
        self.hidden_layer_output = None
        self.hidden_layer_activation = None

    def forward(self, X):
        # propagate some points through the network

        # hidden layer
        # (dot product of data + hidden weights) + hidden bias
        self.hidden_layer_activation = np.dot(X, self.hidden_weights) + self.hidden_bias
        # take relu of result
        self.hidden_layer_output = self.relu(self.hidden_layer_activation)

        # output layer
        # (dot product of hidden layer output + output weights) + output bias
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        # take sigmoid of result
        self.output_layer_output = self.sigmoid(self.output_layer_activation)

        # return the result
        return self.output_layer_output

    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def mean_squared_error(self, y_pred, y_true):
        # mse loss function
        return np.mean((y_pred - y_true) ** 2)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # forward pass
            output = self.forward(X)

            # get the loss (measure progress)
            loss = self.mean_squared_error(output, y)

            # backpropagation below!
            # get number of data points
            N = X.shape[0]

            # average of the output error across sample
            d_output = 2 * (output - y) / N

            # calculate the derivative of the loss with respect to the hidden layer
            # helps distribute errors in output layer back to hidden layer
            d_hidden = np.dot(d_output, self.weights_output.T) * (self.hidden_layer_output > 0)

            # update weights and biases using gradient descent
            self.weights_output -= learning_rate * np.dot(self.hidden_layer_output.T, d_output)
            self.bias_output -= learning_rate * np.sum(d_output, axis=0, keepdims=True)
            self.hidden_weights -= learning_rate * np.dot(X.T, d_hidden)
            self.hidden_bias -= learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

            # track progress through loss
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")


# training data
X = np.array([[1, 1, 1, 0, 0],
              [1, 1, 1, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 1],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 1],
              [0, 0, 1, 1, 0],
              [0, 0, 1, 1, 1],
              [0, 1, 0, 0, 0],
              [0, 1, 0, 0, 1],
              [0, 1, 0, 1, 0],
              [0, 1, 0, 1, 1],
              [0, 1, 1, 0, 0],
              [0, 1, 1, 0, 1],
              [0, 1, 1, 1, 0],
              [0, 1, 1, 1, 1],
              [1, 0, 0, 0, 0],
              [1, 0, 0, 0, 1],
              [1, 0, 0, 1, 0],
              [1, 0, 0, 1, 1],
              [1, 0, 1, 0, 0],
              [1, 0, 1, 0, 1],
              [1, 0, 1, 1, 0],
              [1, 0, 1, 1, 1]])

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

# create the network with input size 5, hidden size 3 and output size 1
nn = NeuralNetwork(inputs=5, hidden_size=3, outputs=1)

# train the network using our data
nn.train(X, y, epochs=1000, learning_rate=0.1)

# test the new inputs
new_input = np.array([[1, 1, 0, 0, 0],
                      [1, 1, 0, 0, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1],
                      [1, 1, 0, 1, 0],
                      [1, 1, 0, 1, 1],
                      [1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1]])

# get predictions based on these new inputs
prediction = nn.forward(new_input)

print("Predictions: ")
print(prediction)
