#!/usr/bin/python

#########################################
# module: cs5600_6600_f23_hw02.py
# Drew Watson
# A02324910
#########################################

import numpy as np
import pickle
from cs5600_6600_f23_hw02_data import *

# sigmoid function and its derivative.
# you'll use them in the training and fitting
# functions below.
def sigmoidf(x):
    return 1 / (1 + np.exp(-1 * x))

def sigmoidf_prime(x):
    return x * (1 - x)

# persists object obj to a file with pickle.dump()
def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        pickle.dump(obj, fp)

# restores the object from a file with pickle.load()
def load(file_name):
    with open(file_name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def build_nn_wmats(mat_dims):
    # mat_dims is an n-tuple which has dimensions of each layer
    weight_matrices = []

    # make a matrix for each "space" between layers
    for i in range(len(mat_dims)-1):
        dim1 = mat_dims[i]
        dim2 = mat_dims[i+1]
        weight_matrix = np.random.rand(dim1, dim2)
        weight_matrices.append(weight_matrix)

    return tuple(weight_matrices)


def build_231_nn():
    return build_nn_wmats((2, 3, 1))

def build_2331_nn():
    return build_nn_wmats((2, 3, 3, 1))

def build_221_nn():
    return build_nn_wmats((2, 2, 1))

def build_838_nn():
    return build_nn_wmats((8, 3, 8))

def build_949_nn():
    return build_nn_wmats((9, 4, 9))

def build_4221_nn():
    return build_nn_wmats((4, 2, 2, 1))

def build_421_nn():
    return build_nn_wmats((4, 2, 1))

def build_121_nn():
    return build_nn_wmats((1, 2, 1))

def build_1221_nn():
    return build_nn_wmats((1, 2, 2, 1))


# Added this build function for bool_3_layer_ann.pck
def build_431_nn():
    return build_nn_wmats((4,3,1))

## Training 3-layer neural net.
## X is the matrix of inputs
## y is the matrix of ground truths.
## build is a nn builder function.
def train_3_layer_nn(numIters, X, y, build):
    ## Build the nn
    weight_matrices = build()
    w1 = weight_matrices[0]
    w2 = weight_matrices[1]

    for i in range(numIters):
        # Find estimated truth
        z2 = np.dot(X,w1)
        a2 = sigmoidf(z2)
        z3 = np.dot(a2,w2)
        yHat = sigmoidf(z3)
        # Back Propagate Errors
        yHat_error = y-yHat
        yHat_delta = yHat_error * sigmoidf_prime(yHat)
        a2_error = np.dot(yHat_delta,w2.T)
        a2_delta = a2_error * sigmoidf_prime(a2)
        w2_delta = np.dot(a2.T,yHat_delta)
        w1_delta = np.dot(X.T,a2_delta)

        # Adjust weights
        w1 = w1 + w1_delta
        w2 = w2 + w2_delta

    ## Return trained matrices
    return tuple([w1, w2])

def train_4_layer_nn(numIters, X, y, build):
    ## Build the nn
    weight_matrices = build()
    w1 = weight_matrices[0]
    w2 = weight_matrices[1]
    w3 = weight_matrices[2]

    for i in range(numIters):
        # Find estimated truth
        z2 = np.dot(X,w1)
        a2 = sigmoidf(z2)
        z3 = np.dot(a2,w2)
        a3 = sigmoidf(z3)
        z4 = np.dot(a3,w3)
        yHat = sigmoidf(z4)
        # Back Propagate Errors
        yHat_error = y-yHat
        yHat_delta = yHat_error * sigmoidf_prime(yHat)
        a3_error = np.dot(yHat_delta,w3.T)
        a3_delta = a3_error * sigmoidf_prime(a3)
        a2_error = np.dot(a3_delta,w2.T)
        a2_delta = a2_error * sigmoidf_prime(a2)
        w3_delta = np.dot(a3.T,yHat_delta)
        w2_delta = np.dot(a2.T,a3_delta)
        w1_delta = np.dot(X.T,a2_delta)
        # Adjust weights
        w1 = w1 + w1_delta
        w2 = w2 + w2_delta
        w3 = w3 + w3_delta

    ## Return trained matrices
    return tuple([w1, w2, w3])

def fit_3_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    w1 = wmats[0]
    w2 = wmats[1]
    # Feed forward
    z2 = np.dot(x,w1)
    a2 = sigmoidf(z2)
    z3 = np.dot(a2,w2)
    yHat = sigmoidf(z3)
    # Threshold
    if thresh_flag:
        for i in range(len(yHat)):
            if yHat[i] <= thresh:
                yHat[i] = 0
            else:
                yHat[i] = 1
    return yHat

def fit_4_layer_nn(x, wmats, thresh=0.4, thresh_flag=False):
    w1 = wmats[0]
    w2 = wmats[1]
    w3 = wmats[2]
    # Feed forward
    z2 = np.dot(x,w1)
    a2 = sigmoidf(z2)
    z3 = np.dot(a2,w2)
    a3 = sigmoidf(z3)
    z4 = np.dot(a3,w3)
    yHat = sigmoidf(z4)
    # Threshold
    if thresh_flag:
        for i in range(len(yHat)):
            if yHat[i] <= thresh:
                yHat[i] = 0
            else:
                yHat[i] = 1
    return yHat

# For bool_3_layer_ann.pck, I used 3 hidden neurons and 1000 iterations.
# For bool_4_layer_ann.pck, I used 2 hidden layers of 2 neurons and 800 iterations