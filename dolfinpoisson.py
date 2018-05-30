

import numpy as np
from flowproblem import FlowProblem
import dolfin as df
import fenics_adjoint as dfa


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
        self.diffusivityFunctionSpace = dfa.FunctionSpace(mesh, 'DG', 0)
        self.sourceTerm = dfa.project(dfa.Constant(0.0), self.solutionFunctionSpace)
        # self.bcPressure = dfa.DirichletBC(self.solutionFunctionSpace, self.pressureField,
        #                       self.pressureBoundary, method='pointwise')
        if not self.pressureBoundary == 'origin':
            self.bcPressure = dfa.DirichletBC(self.solutionFunctionSpace, self.pressureField,
                                              self.pressureBoundary)
        self.bcFlux = df.inner(df.FacetNormal(self.mesh), self.flowField)
        return

    def solvePDE(self, diffusivityFunction):
        # Define variational problem
        u = df.TrialFunction(self.solutionFunctionSpace)
        v = df.TestFunction(self.solutionFunctionSpace)
        a = diffusivityFunction * df.inner(df.grad(v), df.grad(u)) * df.dx
        L = self.sourceTerm * v * df.dx + self.bcFlux * v * df.ds

        # Compute solution
        u = dfa.Function(self.solutionFunctionSpace)
        if not self.pressureBoundary == 'origin':
            dfa.solve(a == L, u, self.bcPressure)
        else:
            dfa.solve(a == L, u)
            origin = np.array([0.0, 0.0], dtype=np.float_)
            u_origin = np.empty(1, dtype=np.float_)
            u.eval(u_origin, origin)
            u.vector()[:] = u.vector()[:] - u_origin

        return u

    def getStiffnessMatrixGradient(self):

        if not len(self.stiffnessMatrixGradient):
            u = df.TrialFunction(self.solutionFunctionSpace)
            v = df.TestFunction(self.solutionFunctionSpace)
            diffusivityFunction = df.Function(self.diffusivityFunctionSpace)
            diffusivityFunction.vector().set_local(np.ones(self.diffusivityFunctionSpace.dim()))
            a = diffusivityFunction * df.inner(df.grad(v), df.grad(u)) * df.dx

            diff_0 = df.Function(self.diffusivityFunctionSpace)
            diff_0_vec = diffusivityFunction.vector().get_local().copy()
            diff_0.vector().set_local(diff_0_vec)
            w = df.TrialFunction(self.solutionFunctionSpace)
            a_0 = diff_0 * df.inner(df.grad(v), df.grad(w)) * df.dx
            K_0 = df.assemble(a_0)
            self.bcPressure.apply(K_0)
            K_0 = K_0.array()

            h = 1e-4
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
                self.stiffnessMatrixGradient.append((K_tmp - K_0)/h)

        return self.stiffnessMatrixGradient
