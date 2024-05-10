import numpy as np

# 1 vector input
# i1
# i2
my_inputs = np.array([2, 3]) # 1x2 vector

# 1 digit output
my_output = 1

# learning rate
a = 0.05


# 2 vector initial weights:
# w1 = 0.11
# w2 = 0.21
#
# w3 = 0.12
# w4 = 0.08

my_layer1_weights = np.array([[0.11, 0.12],     # w1, w3
                              [0.21, 0.08]])    # w2, w4

# Layer1
# L1 = i1 * w1  +  i2 * w2

layer1_output = np.dot(my_inputs,my_layer1_weights)

# Layer2
# w5 = 0.14
# w6 = 0.15

my_layer2_weights = np.array([.14, .15])

# Estimated Output
# Output = L2 = w5 * L1  +  w6 * L2

layer2_output = np.dot(layer1_output,my_layer2_weights)

# Error -- MSE -- Mean Square Error

error = 1/2 * (layer2_output - my_output) ** 2

# Derivative of the error (with respect to weights
# Derivative( Weight - Learning rate * MSE)

delta = layer2_output - my_output


# Backward Pass   (adjust all weights according to gradient of the error)
# new_weights = old_weights - learning_rate * delta * input * previous_layer_old_weights

## Layer 2
my_new_layer2_weights = my_layer2_weights - (a * delta * layer1_output)

## Layer 1
# forward layer is Layer 2
# my_array.reshape(-1,1)  transposes the vector from 1x2 to 2x1 (from python column vector to python row vector)
# my_array.reshape(1,-1)  transposes the vector from 1x2 to 2x1 (from python row vector to python column vector)
# The second transformation is still needed to get thew two dimensions correct for the multiplication
my_new_layer1_weights = my_layer1_weights - (a * delta) * np.dot(my_inputs.reshape(-1,1), my_layer2_weights.reshape(1,-1))


# second pass calculation

layer1_2nd_pass_output = np.dot(my_inputs,my_new_layer1_weights)
layer2_2nd_pass_output = np.dot(layer1_2nd_pass_output,my_new_layer2_weights)
error_2nd_pass = 1/2 * (layer2_2nd_pass_output - my_output) ** 2

print('Expected Output: 1')
print(f'Output on 1st forward pass: {layer2_output}')
print(f'Output on 2nd forward pass: {layer2_2nd_pass_output}')
