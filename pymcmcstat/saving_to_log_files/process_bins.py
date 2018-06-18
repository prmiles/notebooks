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

import time
from pymcmcstat.chain import ChainProcessing as CP
from pymcmcstat.chain.ChainStatistics import chainstats
from pymcmcstat.plotting.MCMCPlotting import Plot

# Initialize classes
MCP = Plot()

# define directory where log files are saved
savedir = 'serial_chain'

# compare reading in binary versus text data
ns = 1
# -------
start = time.time()
for ii in range(ns):
    results = CP.read_in_savedir_files(savedir, extension = 'h5')
    
end = time.time()
binary_time = end - start

# -------
start = time.time()
for ii in range(ns):
    results = CP.read_in_savedir_files(savedir, extension = 'txt')
    
end = time.time()
text_time = end - start
# -------

print('Binary: {} sec\n'.format(binary_time/ns))
print('Text: {} sec\n'.format(text_time/ns))

chain = results['chain']
s2chain = results['s2chain']
sschain = results['sschain']

# define burnin
burnin = 25000
# display chain statistics
stats = chainstats(chain[burnin:,:], returnstats = True)

# generate mcmc plots
#mcpl = mcstat.mcmcplot # initialize plotting methods
#MCP.plot_density_panel(chain[burnin:,:], figsizeinches = (3,3))
MCP.plot_chain_panel(chain[burnin:,:], figsizeinches = (3,3))
MCP.plot_pairwise_correlation_panel(chain[burnin:,:], figsizeinches = (3,3))

CP.print_log_files(savedir)