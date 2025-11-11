"""
    this code is to prove that a combination of neurons can "learn" the patterns for any boolean function,
    that is to say that a single perceptron, for example can represent a single NAND gate, which is universal for
    all boolean functions (-2, -2 weight, 3 bias), but the difference btw them arises but a neural network's ability to 
    learn to create a bit_adder ckt. with just inputs and outputs, 
    like a circuit that is does not require human intervention, but can configure on its own based on 'learning'.
"""

import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 1/(1+e^-x) is the sigmoid squishification fn, ReLU can also be used

def sigmoid_deriv(x):
    return x * (1 - x)

# Training data: inputs and expected outputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Expected outputs: [carry, sum]
y = np.array([
    [0, 0],  # 0 + 0 = carry:0, sum:0
    [0, 1],  # 0 + 1 = carry:1, sum:0
    [0, 1],  # 1 + 0 = carry:1, sum:0
    [1, 0]   # 1 + 1 = carry:0, sum:1
])

# Seed for reproducibility
np.random.seed(24)

# Initialize weights and biases
input_size = 2 # 1 input layer, 2 neuron because 2 bits
hidden_size = 4 # 1 hidden layer, as this is not a linearly seperable fn in a 1-d hyperplane..(heuristics learning later)
output_size = 2 # 1 output layer, 2 neuron because 2 bits (carry + sum)

W1 = np.random.randn(input_size, hidden_size) # random 2rows, 4colums, weights , 1 row, represents connection of any 2 neurons from 1st layer to 2nd
b1 = np.zeros((1, hidden_size)) # 4* 0 bias initially
W2 = np.random.randn(hidden_size, output_size) # random 4rows, 2colums, weights
b2 = np.zeros((1, output_size)) # 2* 0 bias initially

# before training: 
    # print("\n")
    # print(W1)
    # print("\n")
    # print(b1)
    # print("\n")
    # print(W2)
    # print("\n")
    # print(b2)
    # print("\n")


# Training loop
lr = 0.4
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Compute loss (MSE)
    loss = np.mean((y - a2) ** 2)

    # Backpropagation
    error = y - a2
    d2 = error * sigmoid_deriv(a2)
    d1 = np.dot(d2, W2.T) * sigmoid_deriv(a1)

    # Update weights and biases
    W2 += np.dot(a1.T, d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += np.dot(X.T, d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

    # Print loss occasionally
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
# after training: 
    # print("\n")
    # print(W1)
    # print("\n")
    # print(b1)
    # print("\n")
    # print(W2)
    # print("\n")
    # print(b2)
    # print("\n")

# Final predictions
print("\nFinal predictions:")
for i in range(4):
    input_pair = X[i]
    output = sigmoid(np.dot(sigmoid(np.dot(input_pair, W1) + b1), W2) + b2)
    # binary_output = (output > 0.5).astype(int)
    print(f"Input: {input_pair}, Predicted: {output}, Expected: {y[i]}")
