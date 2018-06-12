import numpy as np
import dolfin as df
import matplotlib.pyplot as plt


class ModelParameters:

    def __init__(self):

        self.coarseMesh = df.UnitSquareMesh(2, 2)
        self.coarseSolutionSpace = df.FunctionSpace(self.coarseMesh, 'CG', 1)

        # Function space where pressure is interpolated on
        self.pInterpSpace = df.FunctionSpace(df.UnitSquareMesh(128, 128), 'CG', 1)

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
        self.VRVM_iter = 3

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

    def plot(self, thetaArray, sigmaArray, gammaArray):
        fig = plt.figure(1)

        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(0, 0, 960, 1200)        #half Dell display

        # theta_c
        ax = fig.add_subplot(3, 2, 1)
        ax.clear()
        ax.plot(thetaArray.T)
        plt.axis('tight')
        plt.pause(1e-3)

        # sigma_c
        ax = fig.add_subplot(3, 2, 2)
        ax.clear()
        ax.plot(np.sqrt(sigmaArray.T))
        plt.axis('tight')
        plt.pause(1e-3)

        # gamma
        ax = fig.add_subplot(3, 2, 3)
        ax.clear()
        ax.plot(gammaArray.T)
        plt.axis('tight')
        plt.pause(1e-3)


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



