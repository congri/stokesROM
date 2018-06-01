"""This should be the main module for the python Stokes/Darcy problem.
The code architecture still needs to be specified."""

from stokesdata import StokesData
from reducedordermodel import ReducedOrderModel
from modelparameters import ModelParameters
import numpy as np
import matplotlib.pyplot as plt
import dolfin as df
import fenics_adjoint as dfa
import romtoolbox as rt
import time

df.set_log_level(30)
h = 1e-9

modelParameters = ModelParameters()
stokesData = StokesData()
#stokesData.genData()
stokesData.loadData(('mesh', 'solution'))
stokesData.interpolate('p', modelParameters)

rom = ReducedOrderModel(modelParameters)
x = np.random.normal(-8.9, 0.01, rom.coarseSolver.diffusivityFunctionSpace.dim())
diffusivityFunction = dfa.Function(rom.coarseSolver.diffusivityFunctionSpace)
diffusivityFunction.vector()[:], d_diffusivity = rt.diffusivityTransform(x, 'log', 'backward', return_grad=True)

df.plot(diffusivityFunction)
u_c = rom.coarseSolver.solvePDE(diffusivityFunction)

lgp, d_lgp = rom.log_p_cf(u_c.vector().get_local(), stokesData.p_interp[0].vector().get_local())
print('log_p_cf = ', lgp)
print('d_lgp = ', d_lgp)

adjoints = rom.coarseSolver.getAdjoints(diffusivityFunction, d_lgp)
print('adjoints = ', adjoints)

dK = rom.coarseSolver.getStiffnessMatrixGradient()

start = time.time()
dlgp_dlambda = - adjoints.dot(dK.dot(u_c.vector().get_local()))
end = time.time()
print('Grad computation time = ', end - start)
dlgp_dx = d_diffusivity*dlgp_dlambda
print('dlgp_dlambda = ', dlgp_dlambda)
print('dlgp_dx = ', dlgp_dx)


d_fd = np.zeros(rom.coarseSolver.diffusivityFunctionSpace.dim())
for i in range(0, rom.coarseSolver.diffusivityFunctionSpace.dim()):
    x_fd = x.copy()
    x_fd[i] += h
    diffusivityFunction_fd = dfa.Function(rom.coarseSolver.diffusivityFunctionSpace)
    diffusivityFunction_fd.vector()[:], d_diffusivity = \
        rt.diffusivityTransform(x_fd, 'log', 'backward', return_grad=True)
    u_c_fd = rom.coarseSolver.solvePDE(diffusivityFunction_fd)
    lgp_fd, _ = rom.log_p_cf(u_c_fd.vector().get_local(), stokesData.p_interp[0].vector().get_local())

    d_fd[i] = (lgp_fd - lgp)/h
print('finite difference gradient: ', d_fd)

