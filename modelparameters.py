

import numpy as np
import dolfin as df
import fenics_adjoint as dfa


class ModelParameters:

    def __init__(self):

        self.coarseMesh = df.UnitSquareMesh(2, 2)
        self.coarseSolutionSpace = dfa.FunctionSpace(self.coarseMesh, 'CG', 1)

        # Function space where pressure is interpolated on
        self.pInterpSpace = df.FunctionSpace(df.UnitSquareMesh(128, 128), 'CG', 1)

        # Model hyperparameters
        self.mode = 'local'  # separate theta_c's per macro-cell
        self.priorModel = 'sharedVRVM'

        # log_p_cf parameters
        self.Sinv_vec = 1e-3 * np.ones(self.pInterpSpace.dim())
        # coarse to fine interpolation matrix in log_p_cf
        self.W = computeInterpolationMatrix(self.coarseSolutionSpace, self.pInterpSpace)

        # log_p_c parameters
        sigma_c = np.ones(self.coarseMesh.num_cells())


def computeInterpolationMatrix(fromFunSpace, toFunSpace):
    # Only valid for linear elements?
    W = np.zeros((toFunSpace.dim(), fromFunSpace.dim()))
    for i in range(0, fromFunSpace.dim()):
        f = df.Function(fromFunSpace)
        f.vector().set_local(np.zeros(fromFunSpace.dim()))
        f.vector()[i] = 1.0
        fun = dfa.interpolate(f, toFunSpace)
        W[:, i] = fun.vector().get_local()

    return W



