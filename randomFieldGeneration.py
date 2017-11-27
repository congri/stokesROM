"""This is the class for generating Gaussian random fields with various covariance functions based on spectral
decomposition and Bochner's theorem"""

from numpy import random as rnd
import numpy as np

class RandomField:
    """se:          squared exponential
       ou:          Ornstein-Uhlenbeck
       sinc:        sin(r)/r
       sincSq:      (sin(r)/r)^2
       matern:      Matern"""
    _covarianceFunction = 'se'
    _lengthScale = [.04, .04]
    _params = []    # further covariance function parameters
    _nBasis = 1000  # number of basis functions of truncated random field
    _sigma = 1      # there should be no need to change this parameter



    # Setters and getters
    def set_covarianceFunction(self, cov):
        assert cov == 'se' or cov == 'ou' or cov == 'sinc' or cov == 'sincSq' or cov == 'matern'
        self._covarianceFunction = cov


    def get_covarianceFunction(self): return self._covarianceFunction


    def set_lengthScale(self, l):
        self._lengthScale = l


    def get_lengthScale(self): return self._lengthScale


    def set_params(self, params):
        self._params = params


    def get_params(self): return self._params


    def set_nBasis(self, nBasis):
        self._nBasis = nBasis


    def get_nBasis(self): return self._nBasis




    # Draw realization
    def sample(self):
        # Compute stacked samples of W, see references
        if self._covarianceFunction == 'se':
            W = rnd.multivariate_normal([0, 0], np.diag(np.power(self._lengthScale, -2)), self._nBasis)
        else:
            raise Exception('Unknown covariance type')

        # Compute stacked samples of b, see references
        b = rnd.uniform(0, 2*np.pi, self._nBasis)

        # Draw coefficients gamma, see references
        gamma = rnd.normal(0, 1, [1, self._nBasis])

        def samplefun(x):
            f = np.sqrt((2*self._sigma)/self._nBasis)*(gamma.dot(np.cos(W.dot(x) + b)))
            return f
        return samplefun










