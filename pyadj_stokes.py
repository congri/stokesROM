"""This should be the main module for the python Stokes/Darcy problem.
The code architecture still needs to be specified."""

from stokesdata import StokesData
from reducedordermodel import ReducedOrderModel
from modelparameters import ModelParameters
import numpy as np
import dolfin as df
from multiprocessing import Process
from multiprocessing import Pool
import scipy.optimize as opt
import time



df.set_log_level(30)

modelParams = ModelParameters()
trainingData = StokesData()
# trainingData.genData()
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


rom = ReducedOrderModel(modelParams)

while not converged:

    def neg_log_q(X, Phi, p):
        lg_q, d_lg_q = rom.log_q_n(X, Phi, p)
        return -lg_q, -d_lg_q


    def minimize(args):
        neg_lg_q, x_0, Phi, p_interp = args
        res = opt.minimize(neg_lg_q, x_0, method='BFGS', jac=True, args=(Phi, p_interp))
        return res.x

    x_0 = np.zeros(8)
    args = [(neg_log_q, x_0, trainingData.designMatrix[n],
             trainingData.p_interp[n].vector().get_local()) for n in range(4)]

    max_x = []
    maxmode = 'serial'
    if maxmode == 'serial':
        for arg in args:
            x = minimize(arg)
            max_x.append(x)
    elif maxmode == 'parallel':
        p = Pool(8)
        max_x = p.map(minimize, args)

    if train_iter >= modelParams.max_iterations:
        converged = True
    else:
        train_iter += 1

print('max_x = ', max_x)

_, d_lg_q = rom.log_q_n(max_x[0], trainingData.designMatrix[0], trainingData.p_interp[0].vector().get_local())
_, d_lg_q_0 = rom.log_q_n(np.zeros(8), trainingData.designMatrix[0], trainingData.p_interp[0].vector().get_local())
print('d_log_q_opt = ', d_lg_q)
print('d_log_q_0 = ', d_lg_q_0)








