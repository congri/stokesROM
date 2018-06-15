"""This should be the main module for the python Stokes/Darcy problem.
The code architecture still needs to be specified."""

from stokesdata import StokesData
from reducedordermodel import ReducedOrderModel
from modelparameters import ModelParameters
import numpy as np
import dolfin as df
import multiprocessing as mp
import scipy.optimize as opt
import time
import VI.variationalinference as VI
import matplotlib.pyplot as plt
import romtoolbox as rt

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
sw_decay = .95      # step width decay per epoch
sw_min = 8e-3 * np.ones(2 * modelParams.coarseMesh.num_cells())

# Bring variational distribution params in form for unconstrained optimization
mu_lambda_c = np.zeros(modelParams.coarseMesh.num_cells())
sigma_lambda_c = np.ones(modelParams.coarseMesh.num_cells())
varDistParamsVec = trainingData.samples.size * [np.concatenate((mu_lambda_c, -2 * np.log(sigma_lambda_c)))]


# training phase starts here
converged = False
train_iter = 0
epoch = 0   # one epoch == every data point has been seen once
thetaArray = np.expand_dims(modelParams.theta_c, axis=1)
sigmaArray = np.expand_dims(modelParams.Sigma_c, axis=1)
gammaArray = np.expand_dims(modelParams.gamma, axis=1)
elboList = []


rom = ReducedOrderModel(modelParams, trainingData)

# paramsVec contains [mu, -2*log(sigma)] for every data point
modelParams.paramsVec = trainingData.samples.size * [np.zeros(2*modelParams.coarseMesh.num_cells())]
x_0 = np.zeros(modelParams.coarseMesh.num_cells())      # start point for pre-VI maximization
x_0 = trainingData.samples.size * [x_0]
sq_dist = trainingData.samples.size * [None]

for n in trainingData.samples:
    modelParams.paramsVec[n][0:modelParams.coarseMesh.num_cells()] = -8*np.ones(modelParams.coarseMesh.num_cells())

while not converged:

    # print('Performing pre-VI maximization...')
    #
    # def neg_log_q(X, Phi, p):
    #     lg_q, d_lg_q = rom.log_q_n(X, Phi, p)
    #     return -lg_q, -d_lg_q
    #
    #
    # def minimize_neg_log_q(arg):
    #     neg_lg_q, x_init, Phi, p_interp = arg
    #     res = opt.minimize(neg_lg_q, x_init, method='BFGS', jac=True, args=(Phi, p_interp))
    #     return res.x
    #
    # args = [(neg_log_q, modelParams.paramsVec[n][:modelParams.coarseMesh.num_cells()], trainingData.designMatrix[n],
    #          trainingData.p_interp[n].vector().get_local()) for n in trainingData.samples]
    #
    # max_x = []
    # maxmode = 'parallel'
    # if maxmode == 'serial':
    #     for arg_n in args:
    #         x = minimize_neg_log_q(arg_n)
    #         max_x.append(x)
    # elif maxmode == 'parallel':
    #     parpool = mp.Pool(trainingData.samples.size)
    #     max_x = parpool.map(minimize_neg_log_q, args)
    #     parpool.close()
    #
    # for n in trainingData.samples:
    #     modelParams.paramsVec[n][0:modelParams.coarseMesh.num_cells()] = max_x[n]
    #
    # print('...pre-VI maximization done.')

    # VI starts here
    vimode = 'parallel'
    t_s = time.time()


    def varinf(arg):
        paramsVec_n, Phi_n, pp = arg

        def log_emp_dist(x):
            lg_q_emp, d_lg_q_emp = rom.log_q_n(x, Phi_n, pp)
            return lg_q_emp, d_lg_q_emp

        paramsVec_n = VI.variationalInference(paramsVec_n, log_emp_dist)
        return paramsVec_n


    if vimode == 'serial':
        print('Performing serial variational inference...')
        for n in trainingData.samples:
            modelParams.paramsVec[n] = varinf((modelParams.paramsVec[n].copy(), trainingData.designMatrix[n],
                                       trainingData.p_interp[n].vector().get_local()))

    elif vimode == 'parallel':
        print('Performing variational inference in parallel...')
        args = [(modelParams.paramsVec[n].copy(), trainingData.designMatrix[n].copy(),
                 trainingData.p_interp[n].vector().get_local().copy()) for n in trainingData.samples]

        parpool = mp.Pool(trainingData.samples.size)
        modelParams.paramsVec = parpool.map(varinf, args)
        parpool.close()

    print('...variational inference done.')
    t_e = time.time()

    for n in range(trainingData.samples.size):
        sq_dist[n] = rom.expected_sq_dist(modelParams.paramsVec[n], trainingData.p_interp[n].vector().get_local())

    t_s = time.time()
    elbo, cell_score = rom.M_step(sq_dist)
    elboList.append(elbo)
    t_e = time.time()
    print('M_step time = ', t_e - t_s)
    print('theta_c = ', modelParams.theta_c)
    print('gamma = ', modelParams.gamma[:2])
    print('elbo =', elbo)
    print('cell score = ', cell_score)

    x_c_mode = trainingData.designMatrix[0].dot(modelParams.theta_c)
    effdiff, _ = rt.diffusivityTransform(x_c_mode, 'log', 'backward', return_grad=True)
    print('lambda_c0 = ', effdiff)

    theta_temp = np.expand_dims(modelParams.theta_c, axis=1)
    thetaArray = np.append(thetaArray, theta_temp, axis=1)
    sigma_temp = np.expand_dims(modelParams.Sigma_c, axis=1)
    sigmaArray = np.append(sigmaArray, sigma_temp, axis=1)
    gamma_temp = np.expand_dims(modelParams.gamma, axis=1)
    gammaArray = np.append(gammaArray, gamma_temp, axis=1)

    # plot current parameters
    t_s = time.time()
    if not plt.fignum_exists(1):
        figParams = plt.figure(1, figsize=(960/96, 1200/96), dpi=96)
        # figParams.show()
        for i in range(6):
            figParams.add_subplot(3, 2, i + 1)
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(0, 0, 960, 1200)  # half Dell display
    modelParams.plot(figParams, thetaArray, sigmaArray, gammaArray)
    t_e = time.time()
    print('plot params time = ', t_e - t_s)

    # plot current state
    t_s = time.time()
    if not plt.fignum_exists(2):
        figState = plt.figure(2, figsize=(1920/96, 1200/96), dpi=96)
        # figState.show()
        for i in range(6):
            if (i + 1) % 3 == 0:
                figState.add_subplot(2, 3, i + 1, projection='3d')
            else:
                figState.add_subplot(2, 3, i + 1)
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(50, 50, 1820, 1100)  # full Dell display
    rom.plot_current_state(figState)
    t_e = time.time()
    print('plot state time = ', t_e - t_s)

    t_s = time.time()
    # plot elbo
    if not plt.fignum_exists(3):
        figElbo = plt.figure(3, figsize=(1920/96, 1200/96), dpi=96)
        # figElbo.show()
        ax_elbo = figElbo.add_subplot(1, 1, 1)
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(50, 50, 1820, 1100)  # full Dell display
        ax_elbo.grid(True)
        ax_elbo.set_xlabel('iteration')
        ax_elbo.set_ylabel(r'elbo')
    ax_elbo.plot(np.arange(0, train_iter + 1), elboList, 'kx')
    ax_elbo.set_xlim((0.0, train_iter + 1))
    # figElbo.canvas.draw()
    # figElbo.canvas.flush_events()
    figElbo.savefig('elbo.png')
    # time.sleep(.01)
    t_e = time.time()
    print('plot elbo time = ', t_e - t_s)

    if train_iter >= modelParams.max_iterations:
        converged = True
    else:
        train_iter += 1









