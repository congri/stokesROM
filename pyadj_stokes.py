"""This should be the main module for the python Stokes/Darcy problem.
The code architecture still needs to be specified."""

from stokesdata import StokesData
from reducedordermodel import ReducedOrderModel
from modelparameters import ModelParameters
import numpy as np
import matplotlib.pyplot as plt
import dolfin as df
import romtoolbox as rt
import time
import scipy.sparse as sp


df.set_log_level(30)

modelParams = ModelParameters()
trainingData = StokesData()
#stokesData.genData()
trainingData.loadData(('mesh', 'solution'))
trainingData.interpolate('p', modelParams)
trainingData.shiftData()
trainingData.computeDesignMatrix(modelParams)
trainingData.normalizeDesignMatrix(modelParams)
trainingData.shapeToLocalDesignMatrix()

# theta_c can only be initialized when design matrices exist, i.e. number of features is known
modelParams.theta_c = .0 * np.ones(trainingData.designMatrix[0].shape[1])
modelParams.initHyperparams()

# Step width for ADAM optimization in VI
sw = 1e-1 * np.ones(2 * modelParams.coarseMesh.num_cells())
sw_decay = .95      #step width decay per epoch
sw_min = 8e-3 * np.ones(2 * modelParams.coarseMesh.num_cells())

# Bring variational distribution params in form for unconstrained optimization
mu_lambda_c = np.zeros(modelParams.coarseMesh.num_cells())
sigma_lambda_c = np.ones(modelParams.coarseMesh.num_cells())
varDistParamsVec = trainingData.samples.size * [np.concatenate((mu_lambda_c, -2 * np.log(sigma_lambda_c)))]



# training phase starts here
converged = False
train_iter = 0
epoch = 0   # one epoch == every data point has been seen once
thetaArray = modelParams.theta_c.copy()
sigmaArray = modelParams.sigma_c.copy()
gammaArray = modelParams.gamma.copy()


while not converged:

    if train_iter >= modelParams.max_iterations:
        converged = True
    else:
        train_iter += 1







h = 1e-9
rom = ReducedOrderModel(modelParams)
x = np.random.normal(-8.9, 0.01, rom.coarseSolver.diffusivityFunctionSpace.dim())
diffusivityFunction = df.Function(rom.coarseSolver.diffusivityFunctionSpace)
diffusivityFunction.vector()[:], d_diffusivity = rt.diffusivityTransform(x, 'log', 'backward', return_grad=True)

df.plot(diffusivityFunction)
u_c = rom.coarseSolver.solvePDE(diffusivityFunction)

lgp, d_lgp = rom.log_p_cf(u_c.vector().get_local(), trainingData.p_interp[0].vector().get_local())
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
    diffusivityFunction_fd = df.Function(rom.coarseSolver.diffusivityFunctionSpace)
    diffusivityFunction_fd.vector()[:], d_diffusivity = \
        rt.diffusivityTransform(x_fd, 'log', 'backward', return_grad=True)
    u_c_fd = rom.coarseSolver.solvePDE(diffusivityFunction_fd)
    lgp_fd, _ = rom.log_p_cf(u_c_fd.vector().get_local(), trainingData.p_interp[0].vector().get_local())

    d_fd[i] = (lgp_fd - lgp)/h
print('finite difference gradient: ', d_fd)
print('relative gradient = ', dlgp_dx/d_fd)


