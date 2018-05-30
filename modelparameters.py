

import numpy as np
import dolfin as df
import fenics_adjoint as dfa


class ModelParameters:

    coarseMesh = df.UnitSquareMesh(2, 2)
    coarseSolutionSpace = dfa.FunctionSpace(coarseMesh, 'CG', 1)
    # Function space where pressure is interpolated on
    pInterpSpace = df.FunctionSpace(df.UnitSquareMesh(128, 128), 'CG', 1)
    Sinv_vec = np.ones(pInterpSpace.dim())

    def __init__(self):
        # coarse to fine interpolation matrix
        self.W = computeInterpolationMatrix(self.coarseSolutionSpace, self.pInterpSpace)


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



