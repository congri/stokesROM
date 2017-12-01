"""This is the class for generating Gaussian random fields with various covariance functions based on spectral
decomposition and Bochner's theorem"""

from numpy import random as rnd
import numpy as np

class RandomField:
    """
    se:          squared exponential
    ou:          Ornstein-Uhlenbeck
    sinc:        sin(r)/r
    sincSq:      (sin(r)/r)^2
    matern:      Matern
    """
    covarianceFunction = 'se'
    lengthScale = [.02, .02]
    params = [5.0]    # further covariance function parameters
    nBasis = 1000  # number of basis functions of truncated random field
    sigma = 1.0      # there should be no need to change this parameter


    # Draw realization
    def sample(self):
        # Compute stacked samples of W, see references
        if self.covarianceFunction == 'se':
            # Squared exponential kernel
            W = rnd.multivariate_normal([0, 0], np.diag(np.power(self.lengthScale, -2)), self.nBasis)
        elif self.covarianceFunction == 'ou':
            # Ornstein-Uhlenbeck kernel
            # modulus
            W = rnd.standard_t(2.0, (self.nBasis, 1))/self.lengthScale[0]
            # angle
            phi = 2*np.pi*rnd.uniform(0, 1, (self.nBasis, 1))
            W = np.concatenate((W*np.cos(phi), W*np.sin(phi)), axis=1)
        elif self.covarianceFunction == 'matern':
            # modulus - 'params' is smoothness parameter nu of Matern kernel
            W = rnd.standard_t(self.params[0] + .5, (self.nBasis, 1))/self.lengthScale[0]
            # angle
            phi = 2*np.pi*rnd.uniform(0, 1, (self.nBasis, 1))
            W = np.concatenate((W * np.cos(phi), W * np.sin(phi)), axis=1)
        elif self.covarianceFunction == 'sinc':
            W = (rnd.uniform(0, 1, (self.nBasis, 1)) - .5)/self.lengthScale[0]
            W = np.concatenate((W, (rnd.uniform(0, 1, (self.nBasis, 1)) - .5)/self.lengthScale[1]), axis=1)
        elif self.covarianceFunction == 'sincSq':
            W = rnd.triangular(-1, 0, 1, (self.nBasis, 1))/self.lengthScale[0]
            W = np.concatenate((W, rnd.triangular(-1, 0, 1, (self.nBasis, 1))/self.lengthScale[1]), axis=1)
        else:
            raise Exception('Unknown covariance type')

        # Compute stacked samples of b, see references
        b = rnd.uniform(0, 2*np.pi, self.nBasis)

        # Draw coefficients gamma, see references
        gamma = rnd.normal(0, 1, [1, self.nBasis])

        def samplefun(x):
            f = np.sqrt((2*self.sigma)/self.nBasis)*(gamma.dot(np.cos(W.dot(x) + b)))
            return f
        return samplefun










