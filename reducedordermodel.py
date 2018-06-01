

import numpy as np
from dolfinpoisson import DolfinPoisson


class ReducedOrderModel:

    def __init__(self, ModelParameters):
        self.modelParameters = ModelParameters
        self.coarseSolver = DolfinPoisson(self.modelParameters.coarseMesh, self.modelParameters.coarseSolutionSpace)

    def log_p_cf(self, u_c, u_f_n):
        # Reconstruction distribution
        #   u_f_interp:              model reconstruction
        #   solution_n:     full single solution with index n (velocity and pressure)
        #   d_log_p_d_u_c:  gradient w.r.t. u_c

        diff_n = self.modelParameters.W.dot(u_c) - u_f_n
        log_p = .5 * np.sum(np.log(self.modelParameters.Sinv_vec)) -\
                .5 * self.modelParameters.Sinv_vec.dot(diff_n**2)

        d_log_p_d_u_c = - np.dot(self.modelParameters.Sinv_vec*diff_n, self.modelParameters.W)

        return log_p, d_log_p_d_u_c

    def log_p_c(self, X, Phi, theta_c, sigma_c):
        # Probabilistic map from fine to coarse scale diffusivities
        # X:        transformed coarse scale diffusivity
        # Phi:      design matrices; fine scale diffusivity information is contained here
        # theta_c:  the linear model coefficients
        # sigma_c:  linear model variances

        mu = Phi * theta_c      #mean


# Static functions

