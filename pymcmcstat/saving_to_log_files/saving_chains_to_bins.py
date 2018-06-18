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

#import math
import numpy as np
from pymcmcstat.MCMC import MCMC
from datetime import datetime

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
mcset = MCMC()

# Add data
nds = 100
x = np.linspace(2, 3, num=nds)
y = 2.*x + 3. + 0.1*np.random.standard_normal(x.shape)
mcset.data.add_data_set(x, y)

#datestr = datetime.now().strftime('%Y%m%d_%H%M%S')
#savedir = str('{}_{}'.format(datestr, 'serial_chain'))
savedir = 'serial_chain'
mcset.simulation_options.define_simulation_options(nsimu = int(5.0e4), updatesigma = 1, method = 'dram', 
                                                   savedir = savedir, savesize = 1000, save_to_json = True, 
                                                   verbosity = 0, waitbar = 0, save_to_txt = True, save_to_bin = True)

# update model settings
mcset.model_settings.define_model_settings(sos_function = test_ssfun)

mcset.parameters.add_model_parameter(name = 'm', theta0 = 2., minimum = -10, maximum = np.inf, sample = 1)
mcset.parameters.add_model_parameter(name = 'b', theta0 = -5., minimum = -10, maximum = 100, sample = 1)

mcset.run_simulation()
