#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:34:56 2018

@author: prmiles
"""

import numpy as np
from pymcmcstat.MCMC import MCMC

# define test model function
def test_modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    nrow, ncol = xdata.shape
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
    return y

def test_ssfun(theta, data):
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    # eval model
    ymodel = test_modelfun(xdata, theta)
    # calc sos
    ss = sum((ymodel[:,0] - ydata[:,0])**2)
    return ss

# create data
nds = 100
x = np.linspace(2, 3, num=nds)
y = 2.*x + 3. + 0.1*np.random.standard_normal(x.shape)

# Initialize MCMC object
mcstat = MCMC(rngseed=1234)
# Add data, simulation options, and model settings
mcstat.data.add_data_set(x, y)
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram')
mcstat.model_settings.define_model_settings(sos_function = test_ssfun)
# Add model parameters
mcstat.parameters.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 1)
# run mcmc
mcstat.run_simulation()

# Extract results
results = mcstat.simulation_results.results
chain = results['chain']
# define burnin
burnin = 2000
# display chain statistics
mcstat.chainstats(chain[burnin:,:], results)
print('chain[-1,:] = {}'.format(chain[-1,:]))

# Initialize MCMC object
mcstat = MCMC(rngseed=1234)
# Add data, simulation options, and model settings
mcstat.data.add_data_set(x, y)
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram')
mcstat.model_settings.define_model_settings(sos_function = test_ssfun)
# Add model parameters
mcstat.parameters.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 1)
# run mcmc
mcstat.run_simulation()

# Extract results
results = mcstat.simulation_results.results
chain = results['chain']
# define burnin
burnin = 2000
# display chain statistics
mcstat.chainstats(chain[burnin:,:], results)
print('chain[-1,:] = {}'.format(chain[-1,:]))