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
        self.VRVM_iter = 30

        # log_p_cf parameters
        self.Sinv_vec = 1e-3 * np.ones(self.pInterpSpace.dim())
        # coarse to fine interpolation matrix in log_p_cf
        self.W = computeInterpolationMatrix(self.coarseSolutionSpace, self.pInterpSpace)

        # log_p_c parameters
        self.sigma_c = np.ones(self.coarseMesh.num_cells())

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
            self.gamma = 1e-4 * np.ones_like(self.theta_c)
        elif self.priorModel == 'none':
            self.gamma = None
        else:
            raise ValueError('What prior model for theta_c?')

    def plot_theta_c(self, iteration):
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
        ax.clear()
        ax.plot(iteration*np.ones_like(self.theta_c), self.theta_c, 'x')
        plt.pause(.05)
        plt.show()


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



