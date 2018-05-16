#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  20 11:30:00 2018

@author: prmiles
"""

# import required packages
import numpy as np
from pymcmcstat.MCMC import MCMC
import matplotlib.pyplot as plt

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

# Initialize MCMC object
mcstat = MCMC()

# Add data
nds = 100
x = np.linspace(2, 3, num=nds)
x = x.reshape(nds,1)
m = 2 # slope
b = -3 # offset
noise = 0.1*np.random.standard_normal(x.shape)
y = m*x + b + noise
mcstat.data.add_data_set(x, y)

# initialize parameter array
mcstat.parameters.add_model_parameter(name = 'm', theta0 = 1., minimum = -10, maximum = 10)
mcstat.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100)

# update simulation options
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram', adaptint = 100, verbosity = 1, waitbar = 1)

# update model settings
mcstat.model_settings.define_model_settings(sos_function = test_ssfun)

# Run mcmcrun
mcstat.run_simulation()

# Extract results
results = mcstat.simulation_results.results

chain = results['chain']
s2chain = results['s2chain']
sschain = results['sschain']

names = results['names']

# define burnin
burnin = 2000
# display chain statistics
mcstat.chainstats(chain[burnin:,:], results)
# generate mcmc plots
mcpl = mcstat.mcmcplot # initialize plotting methods
mcpl.plot_density_panel(chain[burnin:,:], names)
mcpl.plot_chain_panel(chain[burnin:,:], names)
mcpl.plot_pairwise_correlation_panel(chain[burnin:,:], names)

# plot data & model
plt.figure()
plt.plot(x,y,'.k')
plt.plot(x, m*x + b, '-r')
model = test_modelfun(x, np.mean(results['chain'],0))
plt.plot(x, model[:,0], '--k') 

# generate prediction intervals
def pred_modelfun(preddata, theta):
    return test_modelfun(preddata.xdata[0], theta)
    
mcstat.PI.setup_prediction_interval_calculation(results = results, data = mcstat.data, modelfunction = pred_modelfun)

mcstat.PI.generate_prediction_intervals()
# plot prediction intervals
mcstat.PI.plot_prediction_intervals(adddata = True)