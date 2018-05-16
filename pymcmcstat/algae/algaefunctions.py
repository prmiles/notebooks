#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 07:33:59 2017

@author: prmiles
"""

#from scipy.integrate import ode
from scipy.integrate import odeint
import numpy as np

def algaess(theta, data):
    # sum-of-squares function for algae example
    ndp, nbatch = data.shape[0]
    time = data.ydata[0][:,0]
    ydata = data.ydata[0][:,1:4]
        
    xdata = data.user_defined_object[0]
    
    # last 3 parameters are the initial states
    y0 = np.array(theta[-3:])

    # evaluate model
    tmodel, ymodel = algaefun(time, theta, y0, xdata)
    ss = np.zeros([nbatch-1])
    
    # calculate sum-of-squares
    for ii in range(nbatch-1):
        ss[ii] = sum((ymodel[:,ii] - ydata[:,ii])**2)

    return ss    

def algaefun(time, theta, y0, xdata):
    """
    Evaluate Ordinary Differential Equation
    """
    
    sol = odeint(algaesys, y0, time, args=(theta, xdata))
    
    return time, sol
    
def algaesys(y, t, theta, xdata):
    """
    Model System
    """
    A = y[0]
    Z = y[1]
    P = y[2]
    
    # control variables are assumed to be saved at each time unit interval
    idx = int(np.ceil(t)) - 1    
    if idx >= 120:
        idx = 119
        
    QpV = xdata[idx,1]
    T = xdata[idx,2]
    Pin = xdata[idx,3]
    
    # model parameters
    mumax = theta[0]
    rhoa = theta[1]
    rhoz = theta[2]
    k = theta[3]
    alpha = theta[4]
    th = theta[5]
    
    mu = mumax*(th**(T-20))*P*((k+P)**(-1))
    
    dotA = (mu - rhoa - QpV - alpha*Z)*A
    dotZ = alpha*Z*A - rhoz*Z
    dotP = -QpV*(P-Pin) + (rhoa-mu)*A + rhoz*Z
    
    ydot = np.array([dotA, dotZ, dotP])
    return ydot