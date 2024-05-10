import numpy as np

# Inputs
initial_inputs = np.array([2, 3])    # 1x2 vector
initial_output = 1                   # 1 digit output

# Parameters
a = 0.05                        # learning rate
epsilon = 0.001                 # acceptable error

# Initial Conditions

# Layer 1
# w1 = 0.11
# w2 = 0.21
#
# w3 = 0.12
# w4 = 0.08

initial_layer1_weights = np.array([[0.11, 0.12],     # w1, w3
                              [0.21, 0.08]])    # w2, w4

# Layer2
# w5 = 0.14
# w6 = 0.15

initial_layer2_weights = np.array([.14, .15])


def nn_run(inputs, output, layer1_weights, layer2_weights):
    # forward propagation
    print('Doing forward propagation')
    layer1_output = np.dot(inputs, layer1_weights)
    layer2_output = np.dot(layer1_output, layer2_weights)
    error = 1 / 2 * (layer2_output - output) ** 2
    delta = layer2_output - output
    print(f'Error is:{error}')
    # back propagation
    print('Doing back propagation')
    my_new_layer2_weights = layer2_weights - (a * delta * layer1_output)
    my_new_layer1_weights = layer1_weights - (a * delta) * np.dot(inputs.reshape(-1, 1),
                                                                     layer2_weights.reshape(1, -1))
    print(f'New weights are:\n'
          f'Layer 1:{my_new_layer1_weights},\n'
          f'Layer 2:{my_new_layer2_weights}')
    return(error, my_new_layer1_weights, my_new_layer2_weights)


# Initialization of error
my_error = 1.0
# Run of initial conditions
(my_error, new_layer1_weights, new_layer2_weights) = nn_run(initial_inputs,initial_output, initial_layer1_weights, initial_layer2_weights)
print(f'Current Error: {my_error}\n')

i = 1
while my_error > epsilon:
    print(f'Interation: {i}')
    (my_error, new_layer1_weights, new_layer2_weights) = nn_run(initial_inputs,initial_output, new_layer1_weights, new_layer2_weights)
    print(f'Current Error: {my_error}\n')
    i +=1

print(f'Process converged after {i} iterations')
print(f'Error is: {my_error}')
print(f'Final weights are:\n'
      f'Layer 1:{new_layer1_weights},\n'
      f'Layer 2:{new_layer2_weights}')

