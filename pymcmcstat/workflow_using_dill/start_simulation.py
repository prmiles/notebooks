#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Demo:
    - Linear Model
    - Start Simulation
    - Save Simulation Using Dill
@author: prmiles
"""

# import required packages
from __future__ import division
import numpy as np
from model_functions import ssfun
from pymcmcstat.MCMC import MCMC
import dill

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
mcstat.model_settings.define_model_settings(sos_function = ssfun)

# Run mcmcrun
mcstat.run_simulation()

# Display statistics
mcstat.chainstats(mcstat.simulation_results.results['chain'], mcstat.simulation_results.results)

# Save simulation object to a file for post processing
def save_dill_object(obj, filename):
    with open(filename, 'wb') as out:
        dill.dump(obj, out)

filename = 'simulation_stage_0'
save_dill_object(mcstat, filename)