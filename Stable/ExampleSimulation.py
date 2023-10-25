# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:21:04 2023

@author: dowel
"""
#%% Example simulation

# The below script will set up the saved files needed to run simple simulations
# and take you through an example. 

# The simulation package is very simple. It initialises a network based upon
# the hemibrain with predicted neurotransmitter identities. You choose a neuron
# type to input a single pulse of activity, which is then transmitted through
# the network through successive iterations. The function effectively sets up a 
# connectivity matrix and does multiple matrix multiplications with an activity 
# vector, defined initially by the neuron types you choose to be active. Like
# in most machine learning artificial neural networks, the output function of 
# units is a ReLu, with the threshold set at zero.The activity vector will 
# change upon successive multiplications and the output at each iteration is saved.

# Obviously this is a very simple way of looking at neural activity. Please feel
# free to make additions to the code set out here to fit your experimental needs.
# For example, you might want to see what the impact of activity crossing  
# retinotopic space is on the central brain. I am currently working on more sophisticated
# simulations like this so please get in touch if you would like some help.

# Anyway enjoy!
#%% Do this before running any code!!!!
# Instructions before running example script
# 1. Download necessary packages see installed_packages.txt
# there are quite a few but this is because I used a standard spyder env and
# added necessary ones.
# 2. Get Neuprint python token from neuprint.janelia.org, copy and paste your
# token ID into line 10 of Simulation functions
# 3. Download neurotransmitter predictions from here: https://storage.googleapis.com/hemibrain/v1.2/hemibrain-v1.2-tbar-neurotransmitters.feather.bz2
# 4. Copy and paste the address of saved neutransmitter .bz2 file into line 25

#%% Run SimulationFunctions

#%% Import packages
from SimulationFunctions import sim_functions as sf
import matplotlib.pyplot as plt
import numpy as np
#%% Run this once
# This will take 15 minutes or so - I would comment the code out when run
S = sf()
S.initialise_NTs()
#%% Example simulation
S = sf()
sim_output = S.run_sim_act(['MBON30','MBON33','MBON08','MBON09'],['MBON.*','FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*'],5)
#%% Plot output
tsim = sim_output['MeanActivityType']

t_norm = np.max(np.abs(tsim),axis=1)
t_norm[0] = 1   
t_norm = t_norm.reshape(-1,1)
tsim = tsim/t_norm

ondx = np.max(np.abs(tsim[:,:]),axis=0)>0.01
plt.figure(figsize=(17,4))
plt.imshow(tsim[:,:],vmax=.5,vmin=-.5,aspect='auto',interpolation='none',cmap='coolwarm')

t_names = sim_output['TypesSmall']
plt.xticks(np.linspace(0,len(t_names)-1,len(t_names)),labels= t_names,rotation=90)
