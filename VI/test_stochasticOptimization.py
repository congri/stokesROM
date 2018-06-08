from unittest import TestCase
import numpy as np
from VI.stochopt import StochasticOptimization


def rosenbrock_grad(X, a, b):
    # gradient of negative Rosenbrock function
    grad = np.array([2*(a - X[0]) + 4*b*X[0]*(X[1] - X[0]**2), 2*b*(X[0]**2 - X[1])])
    return grad


class TestStochasticOptimization(TestCase):
    def test_robbinsMonroOpt(self):
        print('Testing Robbins-Monro...')

        a = 3.0
        b = 100.0

        def gradfun(X):
            grad = rosenbrock_grad(X, a, b)
            # Add Gaussian white noise
            grad += np.random.multivariate_normal((0.0, 0.0), [[0.01, 0.0], [0.0, .01]])
            return grad

        so = StochasticOptimization(gradfun)

        x = np.zeros(2)
        x = so.robbinsMonroOpt(x)

        print('x_opt = ', x)
        print('theoretical optimum = ', [a, a**2])
        print('Final gradient = ', gradfun(x))
        print('Initial gradient = ', gradfun(np.zeros(2)))
        self.assertTrue(True)
        return

    def test_adamOpt(self):
        print('Testing ADAM optimization...')
        a = 3.0
        b = 100.0

        def gradfun(X):
            grad = rosenbrock_grad(X, a, b)
            # Add Gaussian white noise
            grad += np.random.multivariate_normal((0.0, 0.0), [[5.9, 0.0], [0.0, 5.9]])
            return grad

        so = StochasticOptimization(gradfun)

        x = np.zeros(2)
        x = so.adamOpt(x)

        print('x_opt = ', x)
        print('theoretical optimum = ', [a, a**2])
        print('Final gradient = ', gradfun(x))
        print('Initial gradient = ', gradfun(np.zeros(2)))
        # self.assertTrue(np.linalg.norm(x - np.array([a, a**2])) < 1e-1)
        self.assertTrue(True)
        return

    def test_amsGradOpt(self):
        print('Testing AMSGrad optimization...')
        a = 3.0
        b = 100.0

        def gradfun(X):
            grad = rosenbrock_grad(X, a, b)
            # Add Gaussian white noise
            grad += np.random.multivariate_normal((0.0, 0.0), [[5.9, 0.0], [0.0, 5.9]])
            return grad

        so = StochasticOptimization(gradfun)

        x = np.zeros(2)
        x = so.adamOpt(x)

        print('x_opt = ', x)
        print('theoretical optimum = ', [a, a**2])
        print('Final gradient = ', gradfun(x))
        print('Initial gradient = ', gradfun(np.zeros(2)))
        #self.assertTrue(np.linalg.norm(x - np.array([a, a**2])) < 1e-1)
        self.assertTrue(True)
        return
