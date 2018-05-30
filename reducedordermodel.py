

import numpy as np
from dolfinpoisson import DolfinPoisson


class ReducedOrderModel():

    def __init__(self, ModelParameters):
        self.modelParameters = ModelParameters
        self.coarseSolver = DolfinPoisson(self.modelParameters.coarseMesh, self.modelParameters.coarseSolutionSpace)

    def log_p_cf(self, u_c, u_f_n):
        # Reconstruction distribution
        #   u_f_interp:              model reconstruction
        #   solution_n:     full single solution with index n (velocity and pressure)

        diffSq = (self.modelParameters.W.dot(u_c) - u_f_n)**2
        log_p = .5 * np.sum(np.log(self.modelParameters.Sinv_vec))
        log_p -= .5 * self.modelParameters.Sinv_vec.dot(diffSq)

        return log_p


# Static functions

