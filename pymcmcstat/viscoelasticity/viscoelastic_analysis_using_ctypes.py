#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 10:14:07 2018

@author: prmiles
"""

# import required packages
import numpy as np
import scipy.io as sio
from pymcmcstat.MCMC import MCMC
from pymcmcstat.settings.DataStructure import DataStructure
import matplotlib.pyplot as plt
from time import time as timetest
import ctypes
from numpy.ctypeslib import ndpointer

# Load data for VHB 4910
vhbdata = sio.loadmat('vhb4910_data.mat')
time = vhbdata['data']['xdata'][0][0][:,0]
stretch = vhbdata['data']['xdata'][0][0][:,1]
stress = vhbdata['data']['ydata'][0][0][:,0]

nds = len(time)

# Define test parameters
theta0 = {'Gc': 7.5541, 'Ge': 17.69, 'lam_max': 4.8333, 'eta': 33.77, 'gamma': 0.206}
theta0vec = list(theta0.values())

# Define C++ model
## Define hyperelastic model
#hyplib = ctypes.cdll.LoadLibrary('./nonaffine_hyperelastic_model.so')
#nhm = hyplib.nonaffine_hyperelastic_model
#nhm.restype = ndpointer(dtype = ctypes.c_double, shape=(nds,))
#nhm.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_double), ctypes.c_int]

# Define hyperelastic model
def nonaffine_hyperelastic_model(theta, stretch):
    # unpack model parameters
    Gc = theta['Gc']
    Ge = theta['Ge']
    lam_max = theta['lam_max']
    
    # Stretch Invariant
    I1 = stretch**2 + 2/stretch;
    
    # Hydrostratic pressure
    p = (Gc/3/stretch*((9*lam_max**2 - I1)/(3*lam_max**2 - I1))) + Ge/stretch**0.5*(1 - stretch);
    
    # 1st P-K stress in kPa
    Hc = 1/3*Gc*stretch*((9*lam_max**2 - I1)/(3*lam_max**2 - I1));
    He = Ge*(1-1/stretch**2);
    sigma_inf = Hc + He - p/stretch;
    return sigma_inf.reshape([sigma_inf.size, 1])

# Define viscoelastic model
vislib = ctypes.cdll.LoadLibrary('./linear_viscoelastic_model.so')
linear_viscoelastic_model = vislib.linear_viscoelastic_model
linear_viscoelastic_model.restype = ndpointer(dtype = ctypes.c_double, shape=(nds,1))
linear_viscoelastic_model.argtypes = [ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_double), ndpointer(ctypes.c_double), ctypes.c_int]

# Test hyperelastic model evaluation
sigma_inf = nonaffine_hyperelastic_model(theta0, stretch)

# Test viscoelastic model evaluation
q = linear_viscoelastic_model(theta0vec[3], theta0vec[4], stretch, time, nds)

n = 100
st = timetest()
for ii in range(n):
    __ = nonaffine_hyperelastic_model(theta0, stretch)
et = timetest()
print('NHM function evaluation time: {} ms'.format((et - st)/n*1e3))


n = 100
st = timetest()
for ii in range(n):
    __ = linear_viscoelastic_model(theta0vec[3], theta0vec[4], stretch, time, nds)
et = timetest()
print('LVM function evaluation time: {} ms'.format((et - st)/n*1e3))

# Initialize MCMC object
mcstat = MCMC()
# Add data
mcstat.data.add_data_set(x = vhbdata['data']['xdata'][0][0], y = vhbdata['data']['ydata'][0][0])
# Define model parameters
mcstat.parameters.add_model_parameter(name = '$G_c$', theta0 = theta0['Gc'], minimum = 0, sample = False)
mcstat.parameters.add_model_parameter(name = '$G_e$', theta0 = theta0['Ge'], minimum = 0, sample = False)
mcstat.parameters.add_model_parameter(name = '$\lambda_{max}$', theta0 = theta0['lam_max'], minimum = 0, sample = False)
mcstat.parameters.add_model_parameter(name = '$\eta$', theta0 = theta0['eta'], minimum = 0, sample = True)
mcstat.parameters.add_model_parameter(name = '$\gamma$', theta0 = theta0['gamma'], minimum = 0, sample = True)
# Define sum-of-squares function and model settings
def ssfun(t, data):
    # Unpack data structure
    time = data.xdata[0][:,0]
    stretch = data.xdata[0][:,1]
    nds = data.n[0]
    # Assign model parameters
    theta = {'Gc': t[0], 'Ge': t[1], 'lam_max': t[2], 'eta': t[3], 'gamma': t[4]}
    # Evaluate model
    stress_model = nonaffine_hyperelastic_model(theta, stretch) + linear_viscoelastic_model(theta['eta'], theta['gamma'], stretch, time, nds)
    # Calculate sum-of-squares error
    res = data.ydata[0] - stress_model
    ss = np.dot(res.T, res)
#    ss = sum((data.ydata[0] - stress_model)**2)
    return ss

n = 100
st = timetest()
for ii in range(n):
    sstest = ssfun(theta0vec, mcstat.data)
et = timetest()
print('SOS function evaluation time: {} ms'.format((et - st)/n*1e3))
mcstat.model_settings.define_model_settings(sos_function = ssfun)

# Define simulation options
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e3), updatesigma = True)

# Run Simulation
mcstat.run_simulation()

results = mcstat.simulation_results.results
names = results['names']
fullchain = results['chain']
fulls2chain = results['s2chain']
nsimu = results['nsimu']
burnin = int(nsimu/2)
chain = fullchain[burnin:,:]
s2chain = fulls2chain[burnin:,:]

mcstat.chainstats(chain, results)

# plot chain panel
mcstat.mcmcplot.plot_chain_panel(chain, names, figsizeinches=(4,4))
mcstat.mcmcplot.plot_density_panel(chain, names, figsizeinches=(4,4))
mcstat.mcmcplot.plot_pairwise_correlation_panel(chain, names, figsizeinches=(4,4))

# Generate/Plot Prediction/Credible Interval(s)
def predmodelfun(data, t):
    xdata = data.user_defined_object[0]
    theta = {'Gc': t[0], 'Ge': t[1], 'lam_max': t[2], 'eta': t[3], 'gamma': t[4]}
    stress = nonaffine_hyperelastic_model(theta, xdata[:,1]) + linear_viscoelastic_model(theta['eta'], theta['gamma'], xdata[:,1], xdata[:,0], xdata.shape[0])
    return stress

# plot wrt time
pdata = DataStructure()
pdata.add_data_set(x = time, y = stress, user_defined_object = vhbdata['data']['xdata'][0][0])
mcstat.PI.setup_prediction_interval_calculation(results = results, data = pdata, modelfunction = predmodelfun, burnin = burnin)
mcstat.PI.generate_prediction_intervals(nsample = 500, calc_pred_int = True)
# plot prediction intervals
mcstat.PI.plot_prediction_intervals(adddata = True);
plt.xlabel('Time (sec)', fontsize = 22)
plt.xticks(fontsize = 22)
plt.ylabel('Nominal Stress (kPa)', fontsize = 22)
plt.yticks(fontsize = 22)
plt.xlim([0,time[-1]])
plt.ylim([0,240])

# plot wrt stretch
pdata = DataStructure()
sid = np.argmax(stretch)
pdata.add_data_set(x = stretch, y = stress, user_defined_object = vhbdata['data']['xdata'][0][0])
mcstat.PI.setup_prediction_interval_calculation(results = results, data = pdata, modelfunction = predmodelfun, burnin = burnin)
mcstat.PI.generate_prediction_intervals(nsample = 500, calc_pred_int = True)

intervals = mcstat.PI.intervals
intervals_1 = {'credible_intervals': [[intervals['credible_intervals'][0][0][:,0:sid]]],
               'prediction_intervals': [[intervals['prediction_intervals'][0][0][:,0:sid]]]}
stretch_1 = stretch[0:sid]
intervals_2 = {'credible_intervals': [[intervals['credible_intervals'][0][0][:,sid:]]],
               'prediction_intervals': [[intervals['prediction_intervals'][0][0][:,sid:]]]}
stretch_2 = stretch[sid:]
# plot 1st interval
interval_display = {'alpha': 0.75, 'edgecolor': 'none'}
data_display = {'color': 'k', 'linestyle': ':', 'marker': '', 'linewidth': 3}
mcstat.PI.intervals = intervals_1.copy()
mcstat.PI.datapred[0].xdata[0] = stretch_1
mcstat.PI.datapred[0].ydata[0] = stress[0:sid].reshape(stress[0:sid].size,1)
fighandle, axhandle = mcstat.PI.plot_prediction_intervals(adddata = True, addlegend = False, data_display = data_display, interval_display = interval_display);
leghand, leglab = axhandle[0].get_legend_handles_labels()
# plot 2nd interval
mcstat.PI.intervals = intervals_2.copy()
mcstat.PI.datapred[0].xdata[0] = stretch_2
mcstat.PI.datapred[0].ydata[0] = stress[sid:].reshape(stress[sid:].size,1)
mcstat.PI.plot_prediction_intervals(adddata = True, addlegend = False, data_display = data_display, interval_display = interval_display);

plt.xlabel('Stretch (-)', fontsize = 22);
plt.xticks(fontsize = 22);
plt.ylabel('Nominal Stress (kPa)', fontsize = 22);
plt.yticks(fontsize = 22)
plt.xlim([1,6]);
plt.ylim([0,240]);
plt.legend(handles = leghand, labels = leglab);