#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 08:43:44 2018

@author: prmiles
"""

import numpy as np

# define test model function
def modelfun(xdata, theta):
    m = theta[0]
    b = theta[1]
    
    nrow, ncol = xdata.shape
    y = np.zeros([nrow,1])
    y[:,0] = m*xdata.reshape(nrow,) + b
    return y

def ssfun(theta, data):
    
    xdata = data.xdata[0]
    ydata = data.ydata[0]
    
    # eval model
    ymodel = modelfun(xdata, theta)
    
    # calc sos
    ss = sum((ymodel[:,0] - ydata[:,0])**2)
    return ss