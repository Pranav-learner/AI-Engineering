import math
x1=2
x2=3

w1=0.5
w2=0.8

b=1.0

z=w1*x1+w2*x2+b # neuron's inputs are summed , making a neural impulse

a = max(0,z) #RELU for non linear learning

print("z =", z)
print("activation =", a)

def relu(x):
    return max(0, x)

print(relu(-5))
print(relu(0))
print(relu(3))

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

print(sigmoid(-5))
print(sigmoid(0))
print(sigmoid(5))