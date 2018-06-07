"""This should be the main module for the python Stokes/Darcy problem.
The code architecture still needs to be specified."""

from stokesdata import StokesData
from reducedordermodel import ReducedOrderModel
from modelparameters import ModelParameters
import numpy as np
import dolfin as df
import multiprocessing
import scipy.optimize as opt
import time



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

rom = ReducedOrderModel(modelParams)


def log_q(X):
    lg_q, _ = \
        rom.log_q_n(X, trainingData.designMatrix[0], trainingData.p_interp[0].vector().get_local())
    return -lg_q

def d_log_q(X):
    _, d_lg_q = \
        rom.log_q_n(X, trainingData.designMatrix[0], trainingData.p_interp[0].vector().get_local())
    return -d_lg_q

x_0 = np.zeros(8)
initial = log_q(x_0)
print('log_q(x_0) = ', initial)
t_s = time.time()
res = opt.minimize(log_q, x_0, method='BFGS', jac=d_log_q, options={'disp': True})
t_e = time.time()
print('time = ', t_e - t_s)
final = log_q(res['x'])
print('Min. log_q = ', final)
print('final/initial = ', final/initial)






