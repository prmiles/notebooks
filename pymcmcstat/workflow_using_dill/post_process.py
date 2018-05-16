#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Demo:
    - Process Simulation
@author: prmiles
"""

# import required packages
from __future__ import division
from model_functions import modelfun
import numpy as np
import dill

# for graphics
import matplotlib.pyplot as plt

def load_object(filename):
    with open(filename, 'rb') as read:
        out = dill.load(read)
    return out

#mcstat = load_object('simulation_stage_0')
mcstat = load_object('simulation_stage_1')

# Extract results
results = mcstat.simulation_results.results

chain = results['chain']
s2chain = results['s2chain']
sschain = results['sschain']

names = results['names']

# Define burnin
# Typically want to remove first portion of chain, but you need to keep enough
# so you can assess whether or not you have converged.
burnin = 0
# Display chain statistics
mcstat.chainstats(chain[burnin:,:], results)
# Generate mcmc plots
mcpl = mcstat.mcmcplot # initialize plotting methods
mcpl.plot_density_panel(chain[burnin:,:], names)
mcpl.plot_chain_panel(chain[burnin:,:], names)
mcpl.plot_pairwise_correlation_panel(chain[burnin:,:], names)

# Plot data & model
# define test model function
def test_modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    
    nrow, ncol = xdata.shape
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
    return y

m = 2 # true slope
b = -3 # true offset
x = mcstat.data.xdata[0]
y = mcstat.data.ydata[0] # includes noise
plt.figure()
plt.plot(x,y,'.k')
plt.plot(x, m*x + b, '-r')
model = modelfun(x, np.mean(results['chain'],0))
plt.plot(x, model[:,0], '--k') 

# Generate prediction intervals
def pred_modelfun(preddata, theta):
    return modelfun(preddata.xdata[0], theta)
    
mcstat.PI.setup_prediction_interval_calculation(results = results, data = mcstat.data, modelfunction = pred_modelfun)

mcstat.PI.generate_prediction_intervals()
# Plot prediction intervals
mcstat.PI.plot_prediction_intervals(adddata = True)