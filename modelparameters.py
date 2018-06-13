import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
from mpl_toolkits.mplot3d import Axes3D
import time


class ModelParameters:

    def __init__(self):

        self.coarseMesh = df.UnitSquareMesh(2, 2)
        self.coarseSolutionSpace = df.FunctionSpace(self.coarseMesh, 'CG', 1)

        # Function space where pressure is interpolated on
        self.nEl_interp = [128, 128]    # numper of elements in x, y direction of interpolation mesh
        self.interpMesh = df.UnitSquareMesh(self.nEl_interp[0], self.nEl_interp[1])
        self.pInterpSpace = df.FunctionSpace(self.interpMesh, 'CG', 1)

        # Model hyperparameters
        self.mode = 'local'  # separate theta_c's per macro-cell
        self.priorModel = 'sharedVRVM'
        self.variationalDist = 'diagonalGauss'      # variational distribution on lambda_c
        self.VRVM_a = np.finfo(float).eps
        self.VRVM_b = np.finfo(float).eps
        self.VRVM_c = 1e-4
        self.VRVM_d = np.finfo(float).eps
        self.VRVM_e = np.finfo(float).eps
        self.VRVM_f = np.finfo(float).eps
        self.VRVM_iter = 30

        # log_p_cf parameters
        self.Sinv_vec = 1e-3 * np.ones(self.pInterpSpace.dim())
        # coarse to fine interpolation matrix in log_p_cf
        self.W = computeInterpolationMatrix(self.coarseSolutionSpace, self.pInterpSpace)

        # log_p_c parameters
        self.Sigma_c = np.ones(self.coarseMesh.num_cells())

        # Parameters to rescale features
        self.normalization = 'rescale'
        self.featFunMean = None
        self.featFunSqMean = None
        self.featFunMin = None
        self.featFunMax = None

        # Training parameters
        self.max_iterations = 200

    def initHyperparams(self):
        # Initialize hyperparameters gamma. Can only be done after theta_c has been set (because of dimensionality)
        if self.priorModel == 'RVM' or self.priorModel == 'VRVM' or self.priorModel == 'sharedVRVM':
            self.gamma = 1e-6 * np.ones_like(self.theta_c)
        elif self.priorModel == 'none':
            self.gamma = None
        else:
            raise ValueError('What prior model for theta_c?')

    def plot(self, fig, thetaArray, sigmaArray, gammaArray):

        nFeatures = np.size(thetaArray, axis=0)
        if self.priorModel == 'sharedVRVM':
            nFeatures /= self.coarseMesh.num_cells()
            nFeatures = int(nFeatures)

        axes = fig.get_axes()
        # theta_c
        if not axes[0].lines:
            # first iteration - no lines in axes exist
            axes[0].plot(thetaArray.T)
            axes[0].grid(True)
            axes[0].set_xlabel('iteration')
            axes[0].set_ylabel(r'$\theta_c$')
        else:
            iter_index = 0
            for lin in axes[0].lines:
                lin.set_data((np.arange(np.size(thetaArray, axis=1)), thetaArray[iter_index, :]))
                iter_index += 1
            axes[0].relim()  # recompute the data limits
            axes[0].autoscale_view()  # automatic axis scaling
            axes[0].set_ylim(np.amin(thetaArray[:, -1]), np.amax(thetaArray[:, -1]))
            fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)

        # theta_c,i
        if not axes[1].lines:
            # first iteration - no lines in axes exist
            axes[1].plot(np.arange(np.size(thetaArray, axis=0)), thetaArray[:, -1])
            axes[1].grid(True)
            axes[1].autoscale(enable=True, axis='both', tight=True)
            axes[1].set_ylabel(r'$\theta_{c,i}$')
        else:
            lin = axes[1].lines[0]
            lin.set_data((np.arange(np.size(thetaArray, axis=0)), thetaArray[:, -1]))
            axes[1].relim()  # recompute the data limits
            axes[1].autoscale_view()  # automatic axis scaling
            fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)

        # sigma_c
        if not axes[2].lines:
            # first iteration - no lines exist
            axes[2].plot(np.sqrt(sigmaArray.T))
            axes[2].grid(True)
            axes[2].set_xlabel('iteration')
            axes[2].set_yscale('log')
            axes[2].set_ylabel(r'$\sigma_c$')
        else:
            iter_index = 0
            for lin in axes[2].lines:
                lin.set_data((np.arange(np.size(sigmaArray, axis=1)), np.sqrt(sigmaArray[iter_index, :])))
                iter_index += 1
            axes[2].relim()  # recompute the data limits
            axes[2].autoscale_view()  # automatic axis scaling
            fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)

        # sigma_c map
        coords = self.coarseMesh.coordinates()
        tp = axes[3].tripcolor(coords[:, 0], coords[:, 1], self.coarseMesh.cells(), np.sqrt(sigmaArray[:, -1]))
        axes[3].margins(.0, .0)
        axes[3].set_aspect('equal', 'box')
        axes[3].set_title(r'$\sigma_c$')

        pos3 = axes[3].get_position()
        if len(axes) > 6:
            axes[6].remove()
        cbaxes_sigma = fig.add_axes([pos3.x0 + pos3.width + .015, pos3.y0, 0.015, pos3.height])
        plt.colorbar(tp, ax=axes[3], cax=cbaxes_sigma)

        # gamma
        if not axes[4].lines:
            axes[4].plot(gammaArray[:nFeatures, :].T)
            axes[4].grid(True)
            axes[4].set_xlabel('iteration')
            axes[4].set_yscale('log')
            axes[4].set_ylabel(r'$\gamma$')
        else:
            iter_index = 0
            for lin in axes[4].lines:
                lin.set_data((np.arange(np.size(gammaArray, axis=1)), gammaArray[iter_index, :]))
                iter_index += 1
            axes[4].relim()  # recompute the data limits
            axes[4].autoscale_view()  # automatic axis scaling
            fig.canvas.flush_events()  # update the plot and take care of window events (like resizing etc.)

        s = df.Function(self.pInterpSpace)
        s.vector().set_local(np.sqrt(1/self.Sinv_vec))
        plt.sca(axes[5])
        simg = df.plot(s)
        pos6 = axes[5].get_position()
        if len(axes) > 7:
            axes[7].remove()
        cbaxes_s = fig.add_axes([pos6.x0 + pos6.width + .015, pos6.y0, 0.015, pos6.height])
        plt.colorbar(simg, ax=axes[5], cax=cbaxes_s)
        axes[5].set_title(r'$\sigma_{cf}$')

        time.sleep(.01)


def computeInterpolationMatrix(fromFunSpace, toFunSpace):
    # Only valid for linear elements?
    W = np.zeros((toFunSpace.dim(), fromFunSpace.dim()))
    for i in range(0, fromFunSpace.dim()):
        f = df.Function(fromFunSpace)
        f.vector().set_local(np.zeros(fromFunSpace.dim()))
        f.vector()[i] = 1.0
        fun = df.interpolate(f, toFunSpace)
        W[:, i] = fun.vector().get_local()

    return W



