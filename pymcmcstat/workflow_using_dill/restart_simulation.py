#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Demo:
    - Restart Simulation
    - Save Simulation Using Dill
@author: prmiles
"""
# import required packages
from __future__ import division
from model_functions import ssfun
from pymcmcstat.MCMC import MCMC
import dill

# Load results from initial simulation
def load_object(filename):
    with open(filename, 'rb') as read:
        out = dill.load(read)
    return out
mcstat_old = load_object('simulation_stage_0')

# unpack items for restart
results = mcstat_old.simulation_results.results
parameters = mcstat_old.parameters.parameters
x = mcstat_old.data.xdata[0]
y = mcstat_old.data.ydata[0]

# Initialize new MCMC object
mcstat = MCMC()

# add data
mcstat.data.add_data_set(x, y)

# initialize parameter array using previous parameter names and last set of sampled values
mcstat.parameters.add_model_parameter(name = parameters[0]['name'], theta0 = results['theta'][0], minimum = -10, maximum = 10)
mcstat.parameters.add_model_parameter(name = parameters[0]['name'], theta0 = results['theta'][1], minimum = -10, maximum = 100)

# update simulation options - add covariance matrix from previous results
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = 1, method = 'dram', adaptint = 100, verbosity = 1, waitbar = 1, qcov = results['qcov'])

# update model settings
mcstat.model_settings.define_model_settings(sos_function = ssfun)

# Run simulation
mcstat.run_simulation()

# Display statistics
mcstat.chainstats(mcstat.simulation_results.results['chain'], mcstat.simulation_results.results)

# Save simulation object to a file for post processing
def save_object(obj, filename):
    with open(filename, 'wb') as out:
        dill.dump(obj, out)

filename = 'simulation_stage_1'
save_object(mcstat, filename)