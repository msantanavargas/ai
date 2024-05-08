import numpy as np

my_inputs = [2, 3]
my_output = 1
a = 0.05  #learning rate


my_layer1_weights = np.array([[0.11, 0.12],
                              [0.21, 0.08]])

layer1 = np.dot(my_inputs,my_layer1_weights)

my_layer2_weights = [.14, .15]

layer2 = np.dot(layer1,my_layer2_weights)

error = 1/2 * (my_output - layer2) ** 2


#Backward Pass   (move according to gradient)

delta = layer2 - my_output

my_new_layer2_weights = my_layer2_weights - (a * delta * layer1)