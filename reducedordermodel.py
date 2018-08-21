import numpy as np
import scipy as sp
import romtoolbox as rt
from dolfinpoisson import DolfinPoisson
import dolfin as df
import warnings
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time


class ReducedOrderModel:

    def __init__(self, ModelParams, trainingData):
        self.modelParams = ModelParams
        self.trainingData = trainingData
        self.coarseSolver = DolfinPoisson(self.modelParams.coarseMesh, self.modelParams.coarseSolutionSpace)

    def log_p_cf(self, u_c, u_f_n):
        # Reconstruction distribution
        #   u_f_interp:              model reconstruction
        #   solution_n:     full single solution with index n (velocity and pressure)
        #   d_log_p_d_u_c:  gradient w.r.t. u_c

        diff_n = np.array(self.modelParams.W.dot(u_c) - u_f_n)

        log_p = .5 * self.modelParams.sumLogS -\
                .5 * self.modelParams.Sinv_vec.dot(diff_n**2)

        d_log_p_d_u_c = - np.dot(self.modelParams.Sinv_vec*diff_n, self.modelParams.W)

        return log_p, d_log_p_d_u_c

    def log_p_c(self, X, designMatrix_n):
        # Probabilistic map from fine to coarse scale diffusivities
        # X:        transformed coarse scale diffusivity
        # Phi:      design matrices; fine scale diffusivity information is contained here
        # theta_c:  the linear model coefficients
        # sigma_c:  linear model standard deviations

        mu = np.dot(designMatrix_n, self.modelParams.theta_c)      #mean

        # ignore constant log 2pi prefactor
        diff = (mu - X)
        log_p = - .5 * np.sum(np.log(self.modelParams.Sigma_c)) - .5 * np.sum(diff/self.modelParams.Sigma_c)

        # gradient w.r.t. X
        d_log_p_dX = diff/self.modelParams.Sigma_c

        return log_p, d_log_p_dX

    def log_q_n(self, X_n, designMatrix_n, u_f_n):

        lg_p_c_n, d_lg_p_c_n = self.log_p_c(X_n, designMatrix_n)

        diffusivity_vector, d_diffusivity = \
            rt.diffusivityTransform(X_n, 'log', 'backward', return_grad=True)

        stiffnessMatrix = self.coarseSolver.getStiffnessMatrix(diffusivity_vector)
        u_c_n = self.coarseSolver.solvePDE(stiffnessMatrix)

        lg_p_cf_n, d_lg_p_cf_n_d_u_c = self.log_p_cf(u_c_n, u_f_n)

        adjoints = self.coarseSolver.getAdjoints(stiffnessMatrix, d_lg_p_cf_n_d_u_c)
        dK = self.coarseSolver.getStiffnessMatrixGradient()
        d_lg_p_cf_n = - d_diffusivity * adjoints.dot(dK.dot(u_c_n))

        lg_q_n = lg_p_c_n + lg_p_cf_n
        d_lg_q_n = d_lg_p_c_n + d_lg_p_cf_n

        return lg_q_n, d_lg_q_n

    def expected_sq_dist(self, varDistParams_n, u_f_n):
        # compute squared distance between model prediction ond data
        dim = int(.5*varDistParams_n.size)
        inferenceSamples = 100

        if self.modelParams.variationalDist == 'diagonalGauss':
            mu = varDistParams_n[0:dim]
            sigma = np.exp(-.5*varDistParams_n[dim:])
            X = np.random.multivariate_normal(mu, np.diag(sigma), inferenceSamples)
        else:
            raise ValueError('Unknown variational distribution.')

        sqDist = 0.0
        index = 1
        for x in X:
            diffusivity_vector, _ = rt.diffusivityTransform(x, 'log', 'backward', return_grad=True)

            stiffnessMatrix = self.coarseSolver.getStiffnessMatrix(diffusivity_vector)
            u_c = self.coarseSolver.solvePDE(stiffnessMatrix)

            sqDist = (1/index)*((index - 1)*sqDist + np.array(self.modelParams.W.dot(u_c) - u_f_n)**2)
            index += 1

        return sqDist

    def M_step(self, sqDist):

        dim_theta = self.modelParams.theta_c.size
        nElc = self.modelParams.coarseMesh.num_cells()
        XMean = np.empty((self.trainingData.samples.size, nElc))
        XSqMean = np.empty_like(XMean)
        for n in range(self.trainingData.samples.size):
            XMean[n, :] = self.modelParams.paramsVec[n][:nElc]
            XSqMean[n, :] = self.modelParams.paramsVec[n][:nElc]**2 + np.exp(- self.modelParams.paramsVec[n][nElc:])

        if self.modelParams.priorModel == 'VRVM' or self.modelParams.priorModel == 'sharedVRVM':

            # Parameters that do not change when q(lambda_c) is fixed
            a = self.modelParams.VRVM_a + .5
            e = self.modelParams.VRVM_e + .5*self.trainingData.samples.size
            c = self.modelParams.VRVM_c + .5*self.trainingData.samples.size

            sqDistSum = 0.0
            for n in range(self.trainingData.samples.size):
                sqDistSum += sqDist[n]

            f = self.modelParams.VRVM_f + .5 * sqDistSum
            tau_cf = e/f # p_cf precision

            # initialization
            if not (self.modelParams.gamma.size == dim_theta):
                warnings.warn('resizing theta precision parameter gamma')
                self.modelParams.gamma = 1e0 * np.ones(dim_theta)

            gam = self.modelParams.gamma.copy()
            tau_theta = np.diag(gam)     # precision of q(theta_c)
            if not hasattr(self.modelParams, 'Sigma_theta_c'):
                Sigma_theta = np.linalg.inv(tau_theta)
            else:
                Sigma_theta = self.modelParams.Sigma_theta_c

            mu_theta = self.modelParams.theta_c

            for i in range(self.modelParams.VRVM_iter):
                b = self.modelParams.VRVM_b + .5 * (mu_theta**2 + np.diag(Sigma_theta))
                if self.modelParams.priorModel == 'sharedVRVM':
                    b = np.reshape(b, (int(dim_theta/nElc), nElc))
                    b = np.mean(b, axis=1)
                    b = np.tile(b, (1, nElc))
                    b = b.flatten()
                gam = a/b
                d = self.modelParams.VRVM_d + .5 * np.sum(XSqMean, axis=0)
                for n in range(self.trainingData.samples.size):
                    PhiThetaMean_n = self.trainingData.designMatrix[n].dot(mu_theta)
                    d -= XMean[n]*PhiThetaMean_n
                    PhiThetaSq_n = np.diag(PhiThetaMean_n*PhiThetaMean_n.T +
                            self.trainingData.designMatrix[n].dot(Sigma_theta).dot(self.trainingData.designMatrix[n].T))
                    d = d + .5 * PhiThetaSq_n
                tau_c = c/d     # precision of p_c
                sqrt_tau_c = np.sqrt(tau_c)
                tau_theta = np.diag(gam)
                sumPhiTau_cXMean = 0.0
                for n in range(self.trainingData.samples.size):
                    # to ensure pos. def.
                    A = np.diag(sqrt_tau_c).dot(self.trainingData.designMatrix[n])
                    tau_theta = tau_theta + A.T.dot(A)
                    # tau_theta +=
                    # self.trainingData.designMatrix[n].T.dot(diag(tau_c)).dot(self.trainingData.designMatrix[n])
                    sumPhiTau_cXMean += self.trainingData.designMatrix[n].T.dot(np.diag(tau_c)).dot(XMean[n])

                Sigma_theta = np.linalg.inv(tau_theta)
                mu_theta = Sigma_theta.dot(sumPhiTau_cXMean)

            # assign < S >, < Sigma_c >, < theta_c >
            self.modelParams.Sinv_vec = tau_cf
            self.modelParams.sumLogS = - np.sum(np.log(self.modelParams.Sinv_vec))
            self.modelParams.Sigma_c = 1/tau_c
            self.modelParams.theta_c = mu_theta
            self.modelParams.Sigma_theta_c = Sigma_theta
            self.modelParams.gamma = gam
            print('mean_p_cf_std = ', np.sqrt(np.mean(1/self.modelParams.Sinv_vec, axis=0)))

            # THIS IS ONLY VALID FOR FIXED MODEL SETUP
            # Evidence lower bound without constant terms
            # Sigma_lambda_c = XSqMean - XMean. ^ 2; sum_logdet_lambda_c = sum(sum(log(Sigma_lambda_c)));
            # elbo = .5 * logdet(Sigma_theta, 'chol') + ...
            # .5 * sum_logdet_lambda_c - e * sum(log(f)) - ...
            # c * sum(log(d)) - a * sum(log(b));

            # General form of elbo allowing model comparison
            # THIS TYPE OF ELBO IS ONLY VALID FOR FIXED dim(u_f), I.E. INTERPOLATION MODE
            # ONLY VALID IF QoI IS PRESSURE ONLY
            # Short hand notation
            N_dof = self.modelParams.pInterpSpace.dim()
            N = self.trainingData.samples.size
            D_c = self.modelParams.coarseMesh.num_cells()
            aa = self.modelParams.VRVM_a
            bb = self.modelParams.VRVM_b
            cc = self.modelParams.VRVM_c
            dd = self.modelParams.VRVM_d
            ee = self.modelParams.VRVM_e
            ff = self.modelParams.VRVM_f
            D_theta_c = self.modelParams.theta_c.size
            if self.modelParams.priorModel == 'sharedVRVM':
                D_gamma = int(D_theta_c/D_c)    # for shared RVM only!
            else:
                D_gamma = D_theta_c

            Sigma_lambda_c = XSqMean - XMean**2
            sum_logdet_lambda_c = np.sum(np.log(Sigma_lambda_c))

            _, logdet_Sigma_theta = np.linalg.slogdet(Sigma_theta)
            elbo = -.5*N*N_dof*np.log(2*np.pi) + .5*sum_logdet_lambda_c + .5*N*D_c + N_dof*(ee*np.log(ff) +
                np.log(sp.special.gamma(e)) - np.log(sp.special.gamma(ee))) - e*np.sum(np.log(f)) + \
                D_c*(cc*np.log(dd) + np.log(sp.special.gamma(c)) -
                np.log(sp.special.gamma(cc))) - c*np.sum(np.log(d)) + D_gamma*(aa*np.log(bb) +
                np.log(sp.special.gamma(a)) - np.log(sp.special.gamma(aa))) - a*np.sum(np.log(b[:D_gamma])) + \
                .5*logdet_Sigma_theta + .5*D_theta_c
            if self.modelParams.priorModel == 'sharedVRVM':
                gamma_expected = sp.special.digamma(a) - np.log(b)
                elbo += (D_c - 1)*np.sum(.5*gamma_expected - (a/b)*(b - bb))

            cell_score = .5*np.sum(np.log(Sigma_lambda_c), 0) - c*np.log(d)
        else:
            raise ValueError('Unknown prior model.')
        return elbo, cell_score

    def plot_current_state(self, fig):

        axes = fig.get_axes()

        coords = self.modelParams.interpMesh.coordinates()
        diffusivityFunction = df.Function(self.coarseSolver.diffusivityFunctionSpace)

        for n in range(4):

            # data and predictive mode
            p = self.trainingData.p_interp[n].compute_vertex_values()
            x_c_mode = self.trainingData.designMatrix[n].dot(self.modelParams.theta_c)
            diffusivity_vector, _ = rt.diffusivityTransform(x_c_mode, 'log', 'backward', return_grad=True)
            stiffnessMatrix = self.coarseSolver.getStiffnessMatrix(diffusivity_vector)
            u_c = self.coarseSolver.solvePDE(stiffnessMatrix)

            u_reconst = self.modelParams.W.dot(u_c)
            u_reconst_fun = df.Function(self.modelParams.pInterpSpace)
            u_reconst_fun.vector().set_local(u_reconst)
            u_reconst_vtx = u_reconst_fun.compute_vertex_values()
            axes[3*n + 2].cla()
            axes[3*n + 2].plot_trisurf(coords[:, 0], coords[:, 1], u_reconst_vtx)
            axes[3*n + 2].set_zlabel(r'$p$')
            axes[3*n + 2].set_xticks(())
            axes[3*n + 2].set_yticks(())
            axes[3*n + 2].margins(x=.0, y=.0, z=.0)
            axes[3*n + 2].view_init(elev=15, azim=255)
            axes[3*n + 2].plot_trisurf(coords[:, 0], coords[:, 1], p, cmap='inferno')

            # meshes
            if not axes[3*n + 1].get_title():
                plt.sca(axes[3*n + 1])
                axes[3*n + 1].set_title('mesh')
                axes[3*n + 1].set_xticks(())
                axes[3*n + 1].set_yticks(())
                df.plot(self.trainingData.mesh[n], linewidth=1)
                axes[3*n + 1].margins(x=.0, y=.0)

            # mode diffusivities
            plt.sca(axes[3*n])
            diffusivityFunction.vector().set_local(diffusivity_vector)
            pdiff = df.plot(diffusivityFunction, norm=colors.LogNorm(vmin=diffusivity_vector.min(),
                                                                     vmax=diffusivity_vector.max()))

            if not axes[3*n].get_title():
                axes[3 * n].margins(x=.0, y=.0)
                axes[3*n].set_title('eff. diff. mode')
                axes[3*n].set_xticks(())
                axes[3*n].set_yticks(())

            if len(axes) > 12:
                cbaxes = axes[12 + n]
            else:
                pos = axes[3 * n].get_position()
                cbaxes = fig.add_axes([pos.x0 + pos.width - .02, pos.y0, 0.015, pos.height])
            plt.colorbar(pdiff, ax=axes[3*n], cax=cbaxes)

        time.sleep(1e-5)

        return

# Static functions
