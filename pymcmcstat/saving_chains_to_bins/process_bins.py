#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 14:59:22 2018

Description: Load information from bin files and assess state of chain.
Useful when running a simulation on a remote machine.  Can download the latest
chain files, plot the chainpanel, and have a reasonable idea as to whether or
not your chains are converging.

Note, this example is not exhaustive.

@author: prmiles
"""

# import required packages
import numpy as np
from pymcmcstat.MCMCPlotting import MCMCPlotting # for graphics

# load information from bin files
chain = np.loadtxt('bins/chainfile.txt')
sschain = np.loadtxt('bins/sschainfile.txt')
s2chain = np.loadtxt('bins/s2chainfile.txt')

nsimu, npar = chain.shape

# generate generic parameter name set
names = []
for ii in range(npar):
    names.append(str('$p_{}$'.format(ii)))

# Create plotting object
mcpl = MCMCPlotting()
# Plot chainpanel
mcpl.plot_chain_panel(chain[:,:], names)