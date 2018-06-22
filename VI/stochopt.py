"""Stochastic optimization module"""

import numpy as np
import time


class StochasticOptimization:
    def __init__(self, gradfun):
        self.gradfun = gradfun      # function returning the stochastic gradient

        # adam parameters
        self.beta1 = .9
        self.beta2 = .999
        self.epsilon = 1e-4

        self.stepOffset = 10000        # learning rate decay offset parameter
        self.stepWidth = .01
        self.nSamples = 1           # gradient samples per iteration
        self.maxIterations = 1e6
        self.maxCompTime = 5       # maximum optimization time in s

    def adamOpt(self, x):
        # ADAM stochastic maximization
        # x:        initialization point
        converged = False
        curr_step = 1
        stepWidth_stepOffset = self.stepWidth * self.stepOffset

        t_s = time.time()
        while not converged:
            grad = self.gradfun(x)
            if curr_step < 2:
                momentum = 1e-6 * grad
                uncenteredXVariance = grad**2
            else:
                momentum = self.beta1 * momentum + (1 - self.beta1) * grad
            uncenteredXVariance = self.beta2*uncenteredXVariance + (1 - self.beta2) * grad**2

            # update
            x += (stepWidth_stepOffset/(self.stepOffset + curr_step)) *\
                 (1/(np.sqrt(uncenteredXVariance) + self.epsilon)) * momentum

            comp_time = time.time() - t_s
            if comp_time > self.maxCompTime:
                converged = True
                print('Converged because max computation time exceeded')
            elif curr_step > self.maxIterations:
                converged = True
                print('Converged because max number of iterations exceeded')
            else:
                curr_step += 1
                # if curr_step % 5 == 0:
                #     self.nSamples += 1      # Purely heuristic! Remove if unwanted
        return x

    def amsGradOpt(self, x):
        # ADAM stochastic maximization
        # x:        initialization point
        converged = False
        curr_step = 1
        stepWidth_stepOffset = self.stepWidth * self.stepOffset

        t_s = time.time()
        while not converged:
            grad = self.gradfun(x)
            if curr_step < 2:
                momentum = 1e-6 * grad
                uncenteredXVariance = grad**2
                uncenteredXVariance_max = uncenteredXVariance.copy()
            else:
                momentum = self.beta1 * momentum + (1 - self.beta1) * grad
            uncenteredXVariance = self.beta2*uncenteredXVariance + (1 - self.beta2) * grad**2
            uncenteredXVariance_max[uncenteredXVariance_max < uncenteredXVariance] =\
                uncenteredXVariance[uncenteredXVariance_max < uncenteredXVariance]

            # update
            x += (stepWidth_stepOffset/(self.stepOffset + curr_step)) *\
                 (1/(np.sqrt(uncenteredXVariance) + self.epsilon)) * momentum

            comp_time = time.time() - t_s
            if comp_time > self.maxCompTime:
                converged = True
                print('Converged because max computation time exceeded')
            elif curr_step > self.maxIterations:
                converged = True
                print('Converged because max number of iterations exceeded')
            else:
                curr_step += 1
                # if curr_step % 5 == 0:
                #     self.nSamples += 1      # Purely heuristic! Remove if unwanted
        return x

    def robbinsMonroOpt(self, x):
        # Robbins-Monro stochastic maximization
        # x:        starting point of optimization
        converged = False
        curr_step = 1
        stepWidth_stepOffset = self.stepWidth * self.stepOffset

        t_s = time.time()
        while not converged:
            # step delta
            grad = self.gradfun(x)
            delta = (stepWidth_stepOffset/(self.stepOffset + curr_step))*grad
            norm_delta = np.linalg.norm(delta)
            stabilityFactor = 2.0
            if norm_delta > stabilityFactor * np.linalg.norm(x):
                delta = (stabilityFactor/norm_delta) * delta

            x += delta

            comp_time = time.time() - t_s
            if comp_time > self.maxCompTime:
                converged = True
                print('Converged because max computation time exceeded')
            elif curr_step > self.maxIterations:
                converged = True
                print('Converged because max number of iterations exceeded')
            else:
                curr_step += 1
                # if curr_step % 5 == 0:
                #     self.nSamples += 1      # Purely heuristic! Remove if unwanted
        return x


