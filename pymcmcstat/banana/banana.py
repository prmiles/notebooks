#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 13:34:00 2018
% Example from Marko Laine's website: http://helios.fmi.fi/~lainema/mcmc/
% 
% This techncal example constructs a non Gaussian target
% distribution by twisting two first dimensions of Gaussian
% distribution. The Jacobian of the transformation is 1, so it is
% easy to calculate the right probability regions for the banana
% and study different adaptive methods.
%
%%
@author: prmiles
"""

# import required packages
import numpy as np
from pymcmcstat.MCMC import MCMC
from pymcmcstat.EllipseContour import EllipseContour
import matplotlib.pyplot as plt

class Banana_Parameters:
    def __init__(self, rho = 0.9, npar = 12, a = 1, b = 1, mu = None):
        self.rho = rho
        self.a = a
        self.b = b
        
        self.sig = np.eye(npar)
        self.sig[0,1] = rho
        self.sig[1,0] = rho
        self.lam = np.linalg.inv(self.sig)
        
        if mu is None:
            self.mu = np.zeros([npar, 1])
            
npar = 12 # number of model parameters
udobj = Banana_Parameters(npar = npar) # user defined object

# Initialize MCMC object
mcstat = MCMC()
mcstat.data.add_data_set(np.zeros(1),np.zeros(1), user_defined_object = udobj)


# Add model parameters
for ii in range(npar):
    mcstat.parameters.add_model_parameter(name = str('$x_{}$'.format(ii+1)), theta0 = 0.0)

# Define options
mcstat.simulation_options.define_simulation_options(
        nsimu = int(2.0e4), qcov = np.eye(npar)*5, method='dram')

# Define model object
def bananafunction(x, a, b):
    response = x
    response[:,0] = a*x[:,0]
    response[:,1] = x[:,1]*a**(-1) - b*((a*x[:,0])**(2) + a**2)
    return response

def bananainverse(x, a, b):
    response = x
    response[0] = x[0]*a**(-1)
    response[1] = x[1]*a + a*b*(x[0]**2 + a**2)
    return response

def bananass(theta, data):
    x = np.array([theta])
    x = x.reshape(12,1)
    udobj = data.user_defined_object[0]
    lam = udobj.lam
    mu = udobj.mu
    a = udobj.a
    b = udobj.b
    
    baninv = bananainverse(x-mu, a, b)
    
    stage1 = np.matmul(baninv.transpose(),lam)
    stage2 = np.matmul(stage1, baninv)
    
    return stage2
    
mcstat.model_settings.define_model_settings(sos_function = bananass)

# Run simulation
mcstat.run_simulation()

# Extract results
results = mcstat.simulation_results.results
chain = results['chain']
s2chain = results['s2chain']
names = results['names'] # parameter names

# plot chain panel
#plt.rcParams["figure.figsize"] = [12, 12]
mcstat.mcmcplot.plot_chain_panel(chain, names)
plt.savefig('chainpanel.eps', figsize = (8,6), format = 'eps', dpi = 500)

# plot pairwise correlation
mcstat.mcmcplot.plot_pairwise_correlation_panel(chain[:,0:2], names[0:2])

# calculate contours for 50% and 95% critical regions
c50 = 1.3863 # critical values from chisq(2) distribution
c95 = 5.9915

ellipse = EllipseContour()
xe50, ye50 = ellipse.generate_ellipse(udobj.mu, c50*udobj.sig[0:2,0:2])
xe95, ye95 = ellipse.generate_ellipse(udobj.mu, c95*udobj.sig[0:2,0:2])
bxy50 = bananafunction(np.array([xe50, ye50]).T, udobj.a, udobj.b)
bxy95 = bananafunction(np.array([xe95, ye95]).T, udobj.a, udobj.b)

# add countours to pairwise plot
plt.plot(bxy50[:,0], bxy50[:,1], 'k-', LineWidth = 2)
plt.plot(bxy95[:,0], bxy95[:,1], 'k-', LineWidth = 2)
plt.axis('equal')
plt.title('2 first dimensions of the chain with 50% and 95% target contours')
plt.savefig('pairwise.eps', format = 'eps', dpi = 500)