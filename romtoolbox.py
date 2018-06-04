"""Bunch of static functions"""

import pickle
import numpy as np


def diffusivityTransform(x, type='log', dir='forward', limits=np.array([1e-12, 1e12]), **return_grad):
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

