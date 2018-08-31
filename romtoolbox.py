"""Bunch of static functions"""

import matplotlib.pyplot as plt
import numpy as np


def diffusivityTransform(x, type='log', dir='forward', limits=np.array([1e-10, 1e4]), **return_grad):
    # Transformation to positive definite diffusivity from unbounded space, e.g. log diffusivity
    #   'forward': from diffusivity to unbounded quantity
    #   'backward': from unbounded quantity back to diffusivity
    #   return: transformed quantity, derivative of transformation

    if dir == 'forward':
        # from diffusivity to unbounded
        print('not yet implemented, returning 0')
        return 0

    elif dir == 'backward':
        if type == 'logit':
            # Logistic sigmoid transformation
            diffusivity = (limits[1] - limits[0]) / (1 + np.exp(-x)) + limits[0]

            if return_grad:
                d_diffusivity = (limits[1] - limits[0]) * np.exp(-x) / ((1 + np.exp(-x)) ** 2)
                return diffusivity, d_diffusivity
            else:
                return diffusivity
        elif type == 'log':
            diffusivity = np.exp(x)
            diffusivity[diffusivity > limits[1]] = limits[1]
            diffusivity[diffusivity < limits[0]] = limits[0]

            if return_grad:
                return diffusivity, diffusivity
            else:
                return diffusivity

        elif type == 'log_lower_bound':
            diffusivity = np.exp(x) + limits[0]
            diffusivity[diffusivity > limits[1]] = limits[1]

            if return_grad:
                return diffusivity, diffusivity - limits[0]
            else:
                return diffusivity


def finiteDifferenceGradientCheck(function, input):
    # function must have function value and derivative as first and second outputs

    # finite difference epsilon
    epsilon = 1e-6
    tol = 1e-3

    f, d_f = function(input)

    d_f_fd = np.empty_like(input) # finite difference gradient
    for i in range(0, d_f_fd.size):
        input_fd = input.copy()
        input_fd[i] += epsilon

        f_fd, _ = function(input_fd)

        d_f_fd[i] = (f_fd - f)/epsilon


    relgrad = d_f/d_f_fd
    print('gradient/fd gradient = ', relgrad)

    plotit = False
    if plotit:
        plt.plot(relgrad)
        plt.xlabel('component i')
        plt.ylabel('relative gradient')
        plt.show()

    if np.any(abs(relgrad - np.ones_like(relgrad)) > tol):
        raise ValueError('Finite difference gradient check failed.')
        return True
    else:
        return False


def triangrnd(vertices, N=1):
    # uniformly distributed 2d random vectors within the triangle given by vertices
    # vertices: 6d vector with vertex coordinates

    vertices = np.reshape(vertices, (3, 2))

    # Define mapping from reference triangle (0, 0) -- (1, 0) -- (0, 1) to triangle given by vertices
    M = np.transpose(np.vstack((vertices[1] - vertices[0], vertices[2] - vertices[0])))

    # draw random vector in reference triangle
    x1 = np.random.rand(N)
    x2 = (1 - x1)*np.random.rand(N)
    x = np.vstack((x1, x2))

    # map to new triangle
    x_new = M.dot(x)
    x_new[0, :] += vertices[0, 0]
    x_new[1, :] += vertices[0, 1]


    return x_new



