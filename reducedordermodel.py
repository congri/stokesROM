

import numpy as np
import romtoolbox as rt
from dolfinpoisson import DolfinPoisson
import dolfin as df


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

    def log_p_c(self, X, designMatrix_n):
        # Probabilistic map from fine to coarse scale diffusivities
        # X:        transformed coarse scale diffusivity
        # Phi:      design matrices; fine scale diffusivity information is contained here
        # theta_c:  the linear model coefficients
        # sigma_c:  linear model standard deviations

        mu = np.dot(designMatrix_n, self.modelParameters.theta_c)      #mean

        # ignore constant log 2pi prefactor
        diff = (mu - X)
        log_p = - np.sum(np.log(self.modelParameters.sigma_c)) - .5 * np.sum((diff/self.modelParameters.sigma_c)**2)

        # gradient w.r.t. X
        d_log_p_dX = diff/(self.modelParameters.sigma_c**2)

        return log_p, d_log_p_dX

    def log_q_n(self, X_n, designMatrix_n, u_f_n):

        lg_p_c_n, d_lg_p_c_n = self.log_p_c(X_n, designMatrix_n)

        diffusivityFunction = df.Function(self.coarseSolver.diffusivityFunctionSpace)
        diffusivityFunction.vector()[:], d_diffusivity = \
            rt.diffusivityTransform(X_n, 'log', 'backward', return_grad=True)

        u_c_n = self.coarseSolver.solvePDE(diffusivityFunction)
        u_c_n = u_c_n.vector().get_local()

        lg_p_cf_n, d_lg_p_cf_n_d_u_c = self.log_p_cf(u_c_n, u_f_n)
        adjoints = self.coarseSolver.getAdjoints(diffusivityFunction, d_lg_p_cf_n_d_u_c)
        dK = self.coarseSolver.getStiffnessMatrixGradient()
        d_lg_p_cf_n = - d_diffusivity * adjoints.dot(dK.dot(u_c_n))

        lg_q_n = lg_p_c_n + lg_p_cf_n
        d_lg_q_n = d_lg_p_c_n + d_lg_p_cf_n

        return lg_q_n, d_lg_q_n

    def elboGrad(self, log_emp_dist, varDist_mu, varDist_sigma):
        # Samples gradient of ELBO of empirical to variational distribution over lambda_c
        d_mu_mean = .0
        d_sigma_mean = .0
        d_muSq_mean = .0
        d_sigmaSq_mean = .0

        if self.modelParameters.variationalDist == 'diagonalGauss':
            sample = np.random.normal(.0, 1.0, self.modelParameters.coarseMesh.num_cells())

            variationalSample = varDist_mu + varDist_sigma * sample

            _, d_log_empirical = log_emp_dist(variationalSample)

        return

# Static functions

