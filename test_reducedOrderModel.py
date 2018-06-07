from unittest import TestCase
from reducedordermodel import ReducedOrderModel
from modelparameters import ModelParameters
from stokesdata import StokesData
import numpy as np
import romtoolbox as rt
import dolfin as df

df.set_log_level(30)


class TestReducedOrderModel(TestCase):
    def test_log_p_cf(self):
        modelParams = ModelParameters()
        trainingData = StokesData()
        trainingData.loadData(('mesh', 'solution'))
        trainingData.interpolate('p', modelParams)
        trainingData.shiftData()
        trainingData.computeDesignMatrix(modelParams)
        trainingData.normalizeDesignMatrix(modelParams)
        trainingData.shapeToLocalDesignMatrix()

        # theta_c can only be initialized when design matrices exist, i.e. number of features is known
        modelParams.theta_c = np.random.normal(0, 1.0, trainingData.designMatrix[0].shape[1])
        modelParams.initHyperparams()

        rom = ReducedOrderModel(modelParams)

        for n in range(0, 4):
            def log_pcf(u):
                lg_p_cf, d_lg_p_cf = \
                    rom.log_p_cf(u, trainingData.p_interp[n].vector().get_local())
                return lg_p_cf, d_lg_p_cf

            u = np.random.normal(-5.0, 1.0, rom.coarseSolver.solutionFunctionSpace.dim())
            grad_check = rt.finiteDifferenceGradientCheck(log_pcf, u)
            self.assertFalse(grad_check)

    def test_log_p_c(self):
        modelParams = ModelParameters()
        trainingData = StokesData()
        trainingData.loadData(('mesh', 'solution'))
        trainingData.interpolate('p', modelParams)
        trainingData.shiftData()
        trainingData.computeDesignMatrix(modelParams)
        trainingData.normalizeDesignMatrix(modelParams)
        trainingData.shapeToLocalDesignMatrix()

        # theta_c can only be initialized when design matrices exist, i.e. number of features is known
        modelParams.theta_c = np.random.normal(0, 1.0, trainingData.designMatrix[0].shape[1])
        modelParams.initHyperparams()

        rom = ReducedOrderModel(modelParams)

        for n in range(0, 4):
            def log_pc(X):
                lg_p_c, d_lg_p_c = \
                    rom.log_p_c(X, trainingData.designMatrix[n])
                return lg_p_c, d_lg_p_c

            x = np.random.normal(-5.0, 1.0, rom.coarseSolver.diffusivityFunctionSpace.dim())
            grad_check = rt.finiteDifferenceGradientCheck(log_pc, x)
            self.assertFalse(grad_check)

    def test_log_q_n(self):

        modelParams = ModelParameters()
        trainingData = StokesData()
        trainingData.loadData(('mesh', 'solution'))
        trainingData.interpolate('p', modelParams)
        trainingData.shiftData()
        trainingData.computeDesignMatrix(modelParams)
        trainingData.normalizeDesignMatrix(modelParams)
        trainingData.shapeToLocalDesignMatrix()

        # theta_c can only be initialized when design matrices exist, i.e. number of features is known
        modelParams.theta_c = np.random.normal(0, 1.0, trainingData.designMatrix[0].shape[1])
        modelParams.initHyperparams()

        rom = ReducedOrderModel(modelParams)

        for n in range(0, 4):
            def log_q(X):
                lg_q, d_lg_q =\
                    rom.log_q_n(X, trainingData.designMatrix[n], trainingData.p_interp[n].vector().get_local())
                return lg_q, d_lg_q

            x = np.random.normal(-5.0, 1.0, rom.coarseSolver.diffusivityFunctionSpace.dim())
            grad_check = rt.finiteDifferenceGradientCheck(log_q, x)
            self.assertFalse(grad_check)
