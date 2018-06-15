

import numpy as np
from flowproblem import FlowProblem
import dolfin as df
import time


class DolfinPoisson(FlowProblem):
    # Dolfin Darcy solver

    stiffnessMatrixGradient = []

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
        return

    def solvePDE(self, diffusivityFunction):
        # Define variational problem
        u = df.TrialFunction(self.solutionFunctionSpace)
        v = df.TestFunction(self.solutionFunctionSpace)
        a = diffusivityFunction * df.inner(df.grad(v), df.grad(u)) * df.dx
        L = self.sourceTerm * v * df.dx + self.bcFlux * v * df.ds

        # Compute solution
        u = df.Function(self.solutionFunctionSpace)
        ts = time.time()
        df.solve(a == L, u, self.bcPressure)
        print('solve time = ', time.time() - ts)

        # val = np.empty(1, dtype=float)
        # u.eval(val, np.array([.0, .0]))
        # print('u_c origin = ', val)

        return u

    def getAdjoints(self, diffusivityFunction, dJ):
        u = df.TrialFunction(self.solutionFunctionSpace)
        v = df.TestFunction(self.solutionFunctionSpace)
        a = diffusivityFunction * df.inner(df.grad(v), df.grad(u)) * df.dx

        K = df.assemble(a)
        self.bcPressure.apply(K)
        K = K.array()
        adjoints = np.linalg.solve(K.T, dJ)

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

            #Finite difference gradient is independent of h as K depends linear on diffusivity
            h = 1.0
            self.stiffnessMatrixGradient = np.zeros((self.diffusivityFunctionSpace.dim(), K_0.shape[0], K_0.shape[1]))
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

            self.stiffnessMatrixGradient = np.swapaxes(self.stiffnessMatrixGradient, 0, 1)

        return self.stiffnessMatrixGradient
