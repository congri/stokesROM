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

df.set_log_level(30)

stokesData = StokesData()
#stokesData.genData()
stokesData.loadData(('mesh', 'solution'))
stokesData.interpolate('p')

rom = ReducedOrderModel(ModelParameters())
x = np.random.normal(-8.9, 0.01, rom.coarseSolver.diffusivityFunctionSpace.dim())
diffusivityFunction = dfa.Function(rom.coarseSolver.diffusivityFunctionSpace)
diffusivityFunction.vector()[:], d_diffusivity = rt.diffusivityTransform(x, 'log', 'backward', return_grad=True)

df.plot(diffusivityFunction)
u_c = rom.coarseSolver.solvePDE(diffusivityFunction)

lgp = rom.log_p_cf(u_c.vector().get_local(), stokesData.p_interp[0].vector().get_local())
print('log_p_cf = ', lgp)


dK = rom.coarseSolver.getStiffnessMatrixGradient()

print(dK[0])


