import numpy as np

# 1 vector input
# i1
# i2
my_inputs = [2, 3]

# 1 digit output
my_output = 1

# learning rate
0.05


# 2 vector initial weights:
# w1 = 0.11
# w2 = 0.21
#
# w3 = 0.12
# w4 = 0.08

my_layer1_weights = np.array([[0.11, 0.12],
                              [0.21, 0.08]])

# Layer1
# L1 = i1 * w1  +  i2 * w2
layer1 = np.dot(my_inputs,my_layer1_weights)

#Layer2
# w5 = 0.14
# w6 = 0.15
my_layer2_weights = [.14, .15]

layer2 = np.dot(layer1,my_layer2_weights)

error = 1/2 * (my_output - layer2) ** 2


#Backward Pass   (move according to gradient)

delta = layer2 - my_output

my_new_layer2_weights = my_layer2_weights - (a * delta * layer1)