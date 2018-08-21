

import numpy as np
from flowproblem import FlowProblem
import dolfin as df
import scipy as sp
import time


class DolfinPoisson(FlowProblem):
    # Dolfin Darcy solver

    stiffnessMatrixGradient = []
    stiffnessMatrixGradient_swap = []
    stiffnessMatrixConstantTerm = []        # K = K_const + sum_i lambda_i dK/dlambda_i;
                                            # this one is K_const

    # Get boundary condition in dolfin form
    def __init__(self, mesh, solutionSpace):
        # Is this translation of BC's correct?
        # 'method' keyword not working for dfa
        # Create mesh and define function space
        self.mesh = mesh
        self.solutionFunctionSpace = solutionSpace
        self.diffusivityFunctionSpace = df.FunctionSpace(mesh, 'DG', 0)
        self.sourceTerm = df.project(df.Constant(0.0), self.solutionFunctionSpace)
        self.bcPressure = df.DirichletBC(self.solutionFunctionSpace, self.pressureField,
                              self.pressureBoundary, method='pointwise')
        # if not self.pressureBoundary == 'origin':
        #     self.bcPressure = dfa.DirichletBC(self.solutionFunctionSpace, self.pressureField,
        #                                       self.pressureBoundary)
        self.bcFlux = df.inner(df.FacetNormal(self.mesh), self.flowField)
        self.RHS = []
        self.assembled_RHS = []
        self.assembleRHS()
        self.assembled_RHS_local = self.assembled_RHS.get_local()

        return

    def assembleRHS(self):
        v = df.TestFunction(self.solutionFunctionSpace)
        self.RHS = self.sourceTerm * v * df.dx + self.bcFlux * v * df.ds
        self.assembled_RHS = df.assemble(self.RHS)
        self.bcPressure.apply(self.assembled_RHS)

        return

    def getStiffnessMatrix(self, diffusivity_vector):

        if not len(self.stiffnessMatrixGradient):
            self.getStiffnessMatrixGradient()

        stiffnessMatrix = diffusivity_vector.dot(self.stiffnessMatrixGradient_swap) + self.stiffnessMatrixConstantTerm

        return stiffnessMatrix

    def solvePDE(self, stiffnessMatrix):

        # more efficient than fenics solve
        u = np.linalg.solve(stiffnessMatrix, self.assembled_RHS_local)

        return u

    def getAdjoints(self, stiffnessMatrix, dJ):

        adjoints = np.linalg.solve(stiffnessMatrix.T, dJ)

        return adjoints

    def getStiffnessMatrixGradient(self):
        # For computational convenience: zeroth index of dK is zeroth index of K, first index is index
        # of derivative and second index is first index of K
        if not len(self.stiffnessMatrixGradient):
            v = df.TestFunction(self.solutionFunctionSpace)
            diffusivityFunction = df.Function(self.diffusivityFunctionSpace)
            diffusivityFunction.vector().set_local(np.ones(self.diffusivityFunctionSpace.dim()))

            diff_0 = df.Function(self.diffusivityFunctionSpace)
            diff_0_vec = diffusivityFunction.vector().get_local().copy()
            diff_0.vector().set_local(diff_0_vec)
            w = df.TrialFunction(self.solutionFunctionSpace)
            a_0 = diff_0 * df.inner(df.grad(v), df.grad(w)) * df.dx
            K_0 = df.assemble(a_0)
            self.bcPressure.apply(K_0)
            K_0 = K_0.array()

            # Finite difference gradient is independent of h as K depends linear on diffusivity
            h = 1.0
            self.stiffnessMatrixGradient = np.zeros((self.diffusivityFunctionSpace.dim(), K_0.shape[0], K_0.shape[1]))
            self.stiffnessMatrixConstantTerm = K_0.copy()
            for i in range(0, self.diffusivityFunctionSpace.dim()):
                diff_tmp = df.Function(self.diffusivityFunctionSpace)
                diff_tmp_vec = diffusivityFunction.vector().get_local().copy()
                diff_tmp_vec[i] += h
                diff_tmp.vector().set_local(diff_tmp_vec)
                w = df.TrialFunction(self.solutionFunctionSpace)
                a_tmp = diff_tmp * df.inner(df.grad(v), df.grad(w)) * df.dx
                K_tmp = df.assemble(a_tmp)
                self.bcPressure.apply(K_tmp)
                K_tmp = K_tmp.array()
                self.stiffnessMatrixGradient[i, :, :] = (K_tmp - K_0)/h
                self.stiffnessMatrixConstantTerm -= (K_tmp - K_0)/h

            self.stiffnessMatrixGradient_swap = np.swapaxes(self.stiffnessMatrixGradient, 0, 1)

        return self.stiffnessMatrixGradient_swap
