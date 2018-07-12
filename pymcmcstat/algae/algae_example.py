#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:12:31 2017
% Example from Marko Laine's website: http://helios.fmi.fi/~lainema/mcmc/
%
% This is a simplified lake algae dynamics model. We consider
% phytoplankton _A_, zooplankton _Z_ and nutrition _P_
% (eg. phosphorus) available for _A_ in the water. The system is
% affected by the water outflow/inflow _Q_, incoming phosphorus load
% _Pin_ and temperature _T_. It is described as a simple
% predator - pray dynamics between _A_ and _Z_. The growth of _A_ is
% limited by the availability of _P_ and it depends on the water
% temperature _T_. The inflow/outflow _Q_ affects both _A_ and _P_,
% but not _Z_.

%%
Adapted for python by prmiles
"""

# import required packages
import algaefunctions as algfun
import numpy as np
import scipy.io as sio

from pymcmcstat.MCMC import MCMC
# for graphics
from pymcmcstat.plotting import MCMCPlotting

# load Algae data
algaedata = sio.loadmat('algaedata.mat')
# extract dictionary contents
adata = algaedata['data']
tx = adata['xdata'][0][0]
ty = adata['ydata'][0][0]
xlbls = adata['xlabels'][0][0][0]
ylbls = adata['ylabels'][0][0][0]

# initialize MCMC object
mcstat = MCMC()

# initialize data structure 
mcstat.data.add_data_set(x = tx[:,0], y = ty[:,0:4], user_defined_object = tx)

# initialize parameter array
#theta = [0.5, 0.03, 0.1, 10, 0.02, 1.14, 0.77, 1.3, 10]
# add model parameters
mcstat.parameters.add_model_parameter(name = 'mumax', theta0 = 0.5, minimum = 0)
mcstat.parameters.add_model_parameter(name = 'rhoa', theta0 = 0.03, minimum = 0)
mcstat.parameters.add_model_parameter(name = 'rhoz', theta0 = 0.1, minimum = 0)
mcstat.parameters.add_model_parameter(name = 'k', theta0 = 10, minimum = 0)
mcstat.parameters.add_model_parameter(name = 'alpha', theta0 = 0.02, minimum = 0)
mcstat.parameters.add_model_parameter(name = 'th', theta0 = 1.14, minimum = 0, maximum = np.inf, prior_mu = 0.14, prior_sigma = 0.2)
# initial values for the model states
mcstat.parameters.add_model_parameter(name = 'A0', theta0 = 0.77, minimum = 0, maximum = np.inf, prior_mu = 0.77, prior_sigma = 2)
mcstat.parameters.add_model_parameter(name = 'Z0', theta0 = 1.3, minimum = 0, maximum = np.inf, prior_mu = 1.3, prior_sigma = 2)
mcstat.parameters.add_model_parameter(name = 'P0', theta0 = 10, minimum = 0, maximum = np.inf, prior_mu = 10, prior_sigma = 2)

# Generate options
mcstat.simulation_options.define_simulation_options(nsimu = int(1.0e3), updatesigma = 1)

# Define model object:
mcstat.model_settings.define_model_settings(sos_function = algfun.algaess, sigma2 = 0.01**2, S20 = np.array([1,1,2]), N0 = np.array([4,4,4]))

# check model evaluation
theta = [0.5, 0.03, 0.1, 10, 0.02, 1.14, 0.77, 1.3, 10]
ss = algfun.algaess(theta, mcstat.data)

# Run simulation
mcstat.run_simulation()
# Rerun starting from results of previous run
mcstat.simulation_options.nsimu = int(5.0e3)
mcstat.run_simulation(use_previous_results=True)

# extract info from results
results = mcstat.simulation_results.results
chain = results['chain']
s2chain = results['s2chain']
names = results['names'] # parameter names

# display chain stats
mcstat.chainstats(chain, results)

mcpl = MCMCPlotting
# plot chain panel
mcpl.plot_chain_panel(chain, names, figsizeinches = [7, 6])

# plot density panel
mcpl.plot_density_panel(chain, names, figsizeinches = [7, 6])

# pairwise correlation
mcpl.plot_pairwise_correlation_panel(chain, names, figsizeinches = [7, 6])

# ============================
def predmodelfun(data, theta):
    obj = data.user_defined_object[0]
    time = obj[:,0]
    xdata = obj
    # last 3 parameters are the initial states
    y0 = np.array(theta[-3:])
    # evaluate model    
    tmodel, ymodel = algfun.algaefun(time, theta, y0, xdata)
    return ymodel

mcstat.PI.setup_prediction_interval_calculation(results = results, data = mcstat.data, modelfunction = predmodelfun)
mcstat.PI.generate_prediction_intervals(nsample = 500, calc_pred_int = 'on', waitbar = True)
# plot prediction intervals
fighandle, axhandle = mcstat.PI.plot_prediction_intervals(adddata = False, addlegend=False, figsizeinches = [7.5,8])

for ii in range(3):
    axhandle[ii].plot(mcstat.data.ydata[0][:,0], mcstat.data.ydata[0][:,ii+1], 'ro', mfc='none')
    axhandle[ii].set_ylabel('')
    axhandle[ii].set_title(ylbls[ii+1][0])
axhandle[-1].set_xlabel('days')