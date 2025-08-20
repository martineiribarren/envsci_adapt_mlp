

import numpy as np
import pickle

class NeuralNetwork_0hl:
    def __init__(self, input_size, output_size, dyn_learningrate=False):
        """
        Initialize the neural network
        """
        self.input_size = input_size
        self.output_size = output_size
        # Random weights and biases
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)
        self.dyn_learningrate = dyn_learningrate

    def linear(self, x):
        """
        linear activation function
        """
        return x

    def linear_derivative(self, x):
        """
        derivation of linear funktion
        """
        return 1

    def forward(self, input_data):
        """
        Forward Propagation
        """
        self.output = self.linear(np.dot(input_data, self.weights) + self.bias)
        return self.output

    def backward(self, input_data, target):
        """
        Backpropagation
        """
        # Calculate the error
        error = target - self.output

        # Calculate the gradient
        delta = error * self.linear_derivative(self.output)

        # Update weights and bias
        self.weights += self.learning_rate * np.dot(input_data.T, delta)
        self.bias += self.learning_rate * np.sum(delta, axis=0)

    def train(self, input_data, target, epochs, learningrate=0.01, print_output=True):
        """
        Train the network
        """
        self.learning_rate = learningrate
        print_epoch = int(epochs / 20)
        arr_loss = []

        for epoch in range(epochs):
            self.forward(input_data)
            self.backward(input_data, target)
            error = np.mean(np.square(target - self.output))  # Mean Squared Error
            arr_loss.append(error)

            if self.dyn_learningrate:
                if epoch % 100 == 0:
                    self.learning_rate *= 0.9  # Reduce the learning rate every 100 epochs

            if epoch % print_epoch == 0 and print_output:
                print(f"Epoch {epoch + 1}, MSE: {error:.4f}")
            percentage_complete = (epoch / epochs) * 100
            print("Progress: {:.2f}%".format(percentage_complete), end='\r')

        return arr_loss

    def predict(self, input_data):
        """
        Prediction with the trained network
        """
        return self.forward(input_data)

    def save_model(self, file_path):
        """
        Save the trained model to a file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        """
        Load a model from a file.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class NeuralNetwork_2hl:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dyn_learningrate=False):
        """
        Initializes the neural network with two hidden layers.
        """
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.dyn_learningrate = dyn_learningrate

        # Initialize weights and biases randomly
        self.weights1 = np.random.randn(input_size, hidden_size1)
        self.bias1 = np.random.randn(hidden_size1)
        self.weights2 = np.random.randn(hidden_size1, hidden_size2)
        self.bias2 = np.random.randn(hidden_size2)
        self.weights3 = np.random.randn(hidden_size2, output_size)
        self.bias3 = np.random.randn(output_size)

    def relu(self, x):
        """
        ReLU activation function.
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        Derivative of the ReLU function.
        """
        return np.where(x > 0, 1, 0)

    def forward(self, input_data):
        """
        Forward propagation.
        """
        # Layer 1
        self.z1 = np.dot(input_data, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)

        # Layer 2
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.relu(self.z2)

        # Output Layer
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.output = self.z3  # No activation function on the output layer for regression
        return self.output

    def backward(self, input_data, target):
        """
        Backpropagation.
        """
        # Calculate the error
        error = target - self.output

        # Calculate the delta for the output layer
        delta3 = error * 1 
        
        # Calculate the delta for the second hidden layer
        delta2 = delta3.dot(self.weights3.T) * self.relu_derivative(self.z2)

        # Calculate the delta for the first hidden layer
        delta1 = delta2.dot(self.weights2.T) * self.relu_derivative(self.z1)

        # Update weights and biases
        self.weights3 += self.learning_rate * self.a2.T.dot(delta3)
        self.bias3 += self.learning_rate * np.sum(delta3, axis=0)

        self.weights2 += self.learning_rate * self.a1.T.dot(delta2)
        self.bias2 += self.learning_rate * np.sum(delta2, axis=0)

        self.weights1 += self.learning_rate * input_data.T.dot(delta1)
        self.bias1 += self.learning_rate * np.sum(delta1, axis=0)

    def train(self, input_data, target, epochs, learningrate=0.01, print_output=True):
        """
        Train the network.
        """
        self.learning_rate = learningrate
        print_epoch = int(epochs / 20)
        arr_loss = []

        for epoch in range(epochs):
            self.forward(input_data)
            self.backward(input_data, target)
            
            if self.dyn_learningrate:
                if epoch % 100 == 0:
                    self.learning_rate *= 0.9

            error = np.mean(np.square(target - self.output))  # Mean Squared Error
            arr_loss.append(error)
            if epoch % print_epoch == 0 and print_output:
                print(f"Epoch {epoch + 1}, MSE: {error:.4f}")
            percentage_complete = (epoch / epochs) * 100
            print("Progress: {:.2f}%".format(percentage_complete), end='\r')
        return arr_loss

    def predict(self, input_data):
        """
        Prediction with the trained network.
        """
        return self.forward(input_data)

    def save_model(self, file_path):
        """
        Save the trained model to a file.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(file_path):
        """
        Load a model from a file.
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

