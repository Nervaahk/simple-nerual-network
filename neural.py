import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output) 
input_val = np.array([0.7])
losses = []
target = float(input("Please enter a target output value the network will learn"))
dograph = input("Do you want to see the plot? (yes/no): ")
if dograph.lower() == "yes":
    graphyn = True
else:
    graphyn = False
weights = np.random.rand(1)
biases = np.random.rand(1)

hidden_layer1_neurons = 1
hiddenweights1_input = np.random.rand(1, hidden_layer1_neurons)
biases1_hidden = np.random.rand(hidden_layer1_neurons)

hiddenweights1_to_output = np.random.rand(hidden_layer1_neurons, 1)
bias_output = np.random.rand(1) 
learning_rate = 0.1

for i in range(1000000):
    hidden_layer1_input = sigmoid(np.dot(input_val, hiddenweights1_input) + biases1_hidden)
    
    
    z = np.dot(hidden_layer1_input, hiddenweights1_to_output) + bias_output

    output = sigmoid(z)

    error = target - output

    delta = error * sigmoid_derivative(output)
    
    hiddenweights1_to_output += learning_rate * hidden_layer1_input * delta 
    biases += learning_rate * delta

    if i % 1000 == 0:
        loss = error ** 2
        losses.append(loss[0])
        print(f"Iteration {i}, Loss:{loss[0]:.6f}, Output: {output[0]:.6f} ")

print("Training Completed")
print("Final Weights:", weights)
print("Final Biases:", biases)
print("Output after training:", output)

plt.plot(losses)
plt.title("Loss Steps Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss Error")
if graphyn:
    plt.show()


