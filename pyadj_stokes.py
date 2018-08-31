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

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
import pyqtgraph.exporters
import pyqtgraph.opengl as gl

## Switch to using white background and black foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

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
elboArray = np.array([])


rom = ReducedOrderModel(modelParams, trainingData)

# paramsVec contains [mu, -2*log(sigma)] for every data point
modelParams.paramsVec = trainingData.samples.size * [np.concatenate((-8*np.ones(modelParams.coarseMesh.num_cells()),
                                                                    np.zeros(modelParams.coarseMesh.num_cells())))]
sq_dist = trainingData.samples.size * [None]

while not converged:

    print('Performing pre-VI maximization...')

    def neg_log_q(X, Phi, p):
        lg_q, d_lg_q = rom.log_q_n(X, Phi, p)
        return -lg_q, -d_lg_q


    def minimize_neg_log_q(arg):
        neg_lg_q, x_init, Phi, p_interp = arg
        res = opt.minimize(neg_lg_q, x_init, method='BFGS', jac=True, args=(Phi, p_interp))
        return res.x

    args = [(neg_log_q, modelParams.paramsVec[n][:modelParams.coarseMesh.num_cells()], trainingData.designMatrix[n],
             trainingData.p_interp[n].vector().get_local()) for n in trainingData.samples]

    max_x = []
    maxmode = ''        # type any other string to avoid pre-VI maximization
    if maxmode == 'serial':
        for arg_n in args:
            x = minimize_neg_log_q(arg_n)
            max_x.append(x)
    elif maxmode == 'parallel':
        parpool = mp.Pool(trainingData.samples.size)
        max_x = parpool.map(minimize_neg_log_q, args)
        parpool.close()

    if maxmode == 'serial' or maxmode == 'parallel':
        for n in trainingData.samples:
            modelParams.paramsVec[n][0:modelParams.coarseMesh.num_cells()] = max_x[n]

    print('...pre-VI maximization done.')

    # VI starts here
    vimode = 'parallel'
    t_s = time.time()


    def varinf(arg):
        paramsVec_n, Phi_n, pp = arg

        def log_emp_dist(x):
            lg_q_emp, d_lg_q_emp = rom.log_q_n(x, Phi_n, pp)
            return lg_q_emp, d_lg_q_emp

        paramsVec_n = VI.variationalInference(paramsVec_n, log_emp_dist, nSamples=30)
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

    t_e = time.time()
    print('M_step time = ', t_e - t_s)
    print('theta_c = ', modelParams.theta_c)
    print('sigma_c = ', modelParams.Sigma_c)
    print('gamma = ', modelParams.gamma[:rom.modelParams.get_numberOfFeatures()])
    print('elbo =', elbo)
    print('cell score = ', cell_score)
    theta_temp = np.expand_dims(modelParams.theta_c, axis=1)
    thetaArray = np.append(thetaArray, theta_temp, axis=1)
    sigma_temp = np.expand_dims(modelParams.Sigma_c, axis=1)
    sigmaArray = np.append(sigmaArray, sigma_temp, axis=1)
    gamma_temp = np.expand_dims(modelParams.gamma, axis=1)
    gammaArray = np.append(gammaArray, gamma_temp, axis=1)
    elboArray = np.append(elboArray, elbo)

    plt_things = True
    if plt_things:
        # plot current parameters
        t0 = time.time()
        print('Plotting model parameters...')
        # if not plt.fignum_exists(1):
        #     plt.ion()
        #     figParams = plt.figure(1)
        #     figParams.show()
        #     for i in range(6):
        #         figParams.add_subplot(3, 2, i + 1)
        #     mngr = plt.get_current_fig_manager()
        #     mngr.window.setGeometry(0, 0, 960, 1200)  # half Dell display
        # modelParams.plot(figParams, thetaArray, sigmaArray, gammaArray)

        if train_iter == 0:
            app = QtGui.QApplication([])
            win_params = pg.GraphicsWindow(title='params window')  # creates a window
            win_params.resize(960, 1200)        # half Dell display

            # theta vs. iteration
            curve_theta = []
            p_theta = win_params.addPlot(title="Realtime theta", row=0, col=0)
            p_theta.showGrid(x=True, y=True)
            for i in range(len(thetaArray)):
                curve_theta.append(p_theta.plot(pen=pg.mkPen((i, len(thetaArray)), width=2)))

            # theta at iteration i
            p_theta_i = win_params.addPlot(title="Realtime theta_i", row=0, col=1)
            p_theta_i.showGrid(x=True, y=True)
            curve_theta_i = p_theta_i.plot(pen=pg.mkPen('k', width=2))

            # sigma_c vs. iteration
            curve_sigma_c = []
            p_sigma_c = win_params.addPlot(title="Realtime sigma_c", row=1, col=0)
            p_sigma_c.setLogMode(y=True)
            p_sigma_c.showGrid(x=True, y=True)
            for i in range(len(sigmaArray)):
                curve_sigma_c.append(p_sigma_c.plot(pen=pg.mkPen((i, len(sigmaArray)), width=2)))

            # gamma vs. iteration
            nFeatures = np.size(thetaArray, axis=0)
            if modelParams.priorModel == 'sharedVRVM':
                nFeatures /= modelParams.coarseMesh.num_cells()
                nFeatures = int(nFeatures)

            curve_gamma = []
            p_gamma = win_params.addPlot(title="Realtime gamma", row=2, col=0)
            p_gamma.setLogMode(y=True)
            p_gamma.showGrid(x=True, y=True)
            for i in range(nFeatures):
                curve_gamma.append(p_gamma.plot(pen=pg.mkPen((i, nFeatures), width=2)))

        # theta vs. iteration
        for i in range(len(thetaArray)):
            curve_theta[i].setData(thetaArray[i])

        # theta at iteration i
        curve_theta_i.setData(thetaArray[:, -1])

        # sigma_c vs. iteration
        for i in range(len(sigmaArray)):
            curve_sigma_c[i].setData(sigmaArray[i])

        # gamma vs. iteration
        for i in range(nFeatures):
            curve_gamma[i].setData(gammaArray[i])

        exporter = pg.exporters.SVGExporter(p_gamma.scene())
        exporter.export('params_pyqtgraph.svg')
        print('...model parameters plotted.')
        t1 = time.time()
        print('plot time == ', t1 - t0)

        # plot current state
        t2 = time.time()
        print('Plotting current reconstruction state...')
        if not plt.fignum_exists(2):
            figState = plt.figure(2)
            figState.show()
            for i in range(12):
                if (i + 1) % 3 == 0:
                    figState.add_subplot(4, 3, i + 1, projection='3d')
                else:
                    figState.add_subplot(4, 3, i + 1)
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(50, 50, 1820, 1100)  # full Dell display
        rom.plot_current_state(figState)
        print('...reconstruction plotted.')
        t3 = time.time()
        print('plot time == ', t3 - t2)

        # elbo plot
        t4 = time.time()
        print('Plotting elbo...')
        if train_iter == 0:
            # app = QtGui.QApplication([])
            win_elbo = pg.GraphicsWindow(title='Elbo window')  # creates a window
            p_elbo = win_elbo.addPlot(title="Realtime elbo")  # creates empty space for the plot in the window
            p_elbo.showGrid(x=True, y=True)
            curve_elbo = p_elbo.plot(pen=pg.mkPen('k', width=2))  # create an empty "plot" (a curve to plot)

        curve_elbo.setData(np.linspace(0, train_iter, train_iter + 1), elboArray)
        # app.processEvents()
        exporter = pg.exporters.SVGExporter(p_elbo.scene())
        # exporter.params.param('width').setValue(640, blockSignal=exporter.widthChanged)
        # exporter.params.param('height').setValue(480, blockSignal=exporter.heightChanged)
        exporter.export('elbo_pyqtgraph.svg')
        time.sleep(1e-1)
        print('...elbo plotted.')
        t5 = time.time()
        print('plot time == ', t5 - t4)

    if train_iter >= modelParams.max_iterations:
        converged = True
    else:
        train_iter += 1


plt.savefig('./current_state.png')




