'''Variational inference functions'''

import numpy as np
import VI.stochopt as so


def elboGrad(paramsVec, log_emp_dist, nSamples=5, varDist='diagonalGauss'):
    # Samples gradient of ELBO of empirical to variational distribution over lambda_c
    d_mu_mean = .0
    d_sigma_mean = .0
    d_muSq_mean = .0
    d_sigmaSq_mean = .0

    if varDist == 'diagonalGauss':
        dim = int(paramsVec.size/2)
        # backtransformation
        varDist_mu = paramsVec[0:dim]
        varDist_sigma = np.exp(-.5 * paramsVec[dim:])
        for i in range(nSamples):
            sample = np.random.normal(.0, 1.0, dim)

            # transform standard normal sample to sample of variational distribution
            variationalSample = varDist_mu + varDist_sigma * sample

            # gradient w.r.t. dependent variable x
            _, d_log_empirical = log_emp_dist(variationalSample)

            # mean gradient w.r.t. mu; d/dmu = (dx/dmu)*d/dx, x = mu + sigma * sample
            # --> d/dmu = d/dx
            d_mu_mean = (1.0/(i + 1.0))*(i * d_mu_mean + d_log_empirical)

            # mean gradient w.r.t. d/d_sigma_k; d/dsigma = (dX/dsigma)*d/dX,
            # X = mu + sigma*sample --> d/dsigma = sample*(d/dX)
            # second term is due to gradient of variational dist (given analytically)
            d_sigma_mean = (1.0/(i + 1.0)) * \
                           (i * d_sigma_mean + d_log_empirical *sample + 1/ varDist_sigma)

            d_muSq_mean = (1.0 / (i + 1.0)) * (i * d_muSq_mean + d_log_empirical ** 2)

            d_sigmaSq_mean = \
                (1.0 / (i + 1.0)) * (i * d_sigmaSq_mean + (d_log_empirical * sample + 1 / varDist_sigma) ** 2)

        # Transformation d / dsigma --> d / dlog(sigma ^ -2)
        d_logSigma_Minus2mean = -.5 * (d_sigma_mean * varDist_sigma)
        ELBOgrad = np.concatenate((d_mu_mean, d_logSigma_Minus2mean))

        d_muErr = np.sqrt(abs(d_muSq_mean - d_mu_mean ** 2)) / np.sqrt(nSamples)
        # error w.r.t.d / d_sigma_k
        d_sigmaErr = \
            np.sqrt(.25 * (varDist_sigma ** 2) * d_sigmaSq_mean - d_logSigma_Minus2mean ** 2) / np.sqrt(nSamples)
        ELBOgradErr = np.concatenate((d_muErr, d_sigmaErr))

    return ELBOgrad, ELBOgradErr


def variationalInference(paramsVec, log_emp_dist, varDist='diagonalGauss', method='AMSGrad', nSamples=5):

    def gradfun(x):
        grad, _ = elboGrad(x, log_emp_dist, nSamples, varDist)
        return grad

    sopt = so.StochasticOptimization(gradfun)
    if method == 'AMSGrad':
        paramsVec = sopt.amsGradOpt(paramsVec)
    elif method == 'ADAM':
        paramsVec = sopt.adamOpt(paramsVec)
    elif method == 'RobbinsMonro':
        paramsVec = sopt.robbinsMonroOpt(paramsVec)
    else:
        raise ValueError('Unknown stochastic optimization method for variational inference')

    return paramsVec

