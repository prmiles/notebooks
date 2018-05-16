#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 14:25:17 2018

@author: prmiles
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 12:12:31 2017

@author: prmiles
"""

# import required packages
import numpy as np
from pymcmcstat.MCMC import MCMC

# define test model function
def test_modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    
    nrow, ncol = xdata.shape
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
#    y[:,1] = m*(xdata.reshape(nrow,))**2 + b
    return y

def test_ssfun(theta, data):
    
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    
    # eval model
    ymodel = test_modelfun(xdata, theta)
    
    # calc sos
    ss1 = sum((ymodel[:,0] - ydata[:,0])**2)
#    ss2 = sum((ymodel[:,1] - ydata[:,0])**2)
    return ss1#np.array([ss1, ss2])

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
mcstat.simulation_options.define_simulation_options(nsimu = int(5.0e4), updatesigma = 1, method = 'dram', adaptint = 100, verbosity = 1, waitbar = 1, save_to_bin = True, savesize = 1000, savedir = 'bins')

# update model settings
mcstat.model_settings.define_model_settings(sos_function = test_ssfun)

# Run mcmcrun
mcstat.run_simulation()
