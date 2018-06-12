from unittest import TestCase
import numpy as np
import VI.variationalinference as VI


class TestVariationalInference(TestCase):
    def test_variationalInference(self):

        mu_test = np.array([18.0, 6.0])
        sigma_test = np.array([4.0, 5.0])
        paramsVec_true = np.concatenate((mu_test, -2*np.log(sigma_test)))

        print('Variational inference unit test: fit diagonal Gaussian to diagonal Gaussian with mu = ',
              mu_test, ' and sigma = ', sigma_test)

        def log_emp_dist(x):
            # 1d Gaussian for testing

            lg_emp_dist = None      #dummy
            d_lg_emp_dist = (mu_test - x)/(sigma_test**2)

            return lg_emp_dist, d_lg_emp_dist

        paramsVec = np.zeros_like(paramsVec_true)        #start from standard normal distribution

        paramsVec = VI.variationalInference(paramsVec, log_emp_dist, 'diagonalGauss', 'AMSGrad')

        print('mu_VI = ', paramsVec[0])
        print('sigma_VI = ', np.exp(-.5 * paramsVec[1]))

        print('paramsVec = ', paramsVec)
        print('paramsVec_true = ', paramsVec_true)
        relParams = paramsVec/paramsVec_true
        print('rel. params = ', relParams)

        mean_rel = np.mean(relParams)

        self.assertAlmostEqual(mean_rel, 1.0, places=1)

