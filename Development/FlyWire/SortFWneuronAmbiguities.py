# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:39:05 2024

@author: dowel
"""

#%% Script aim
# Aim is to resolve ambiguities in flywire neuron assignment by comparing pre and 
# post-synaptic connectivity to Neuprint neurons.

# The metrics outputted are the pearson's and spearman's correlation coefficients

# For now it is probably best to test on a case by case basis, since the correlations
# will be affected by the number of pre and post synaptic partners that have been
# sucessfully identified across Neuprint and flywire datasets

# There is the option of using reassigned values by toggling classifications Charlie
# when setting up.

# Only classifications Charlie will be updated with your new neuron assignment.
# Therefore you should use this classification system when doing other analyses

#%%
from Stable.CorrectFlyWire import fw_corrections
import matplotlib.pyplot as plt
#%% Example %%%%%%%%

# %% Initialise and get predictions
NP_neuron = 'FB4E'
fw = fw_corrections(classtype='Charlie')
predictions = fw.allocate_by_connections(NP_neuron)   

# plot predictions
pcorr = predictions['Corr_pearson']
pcorr_top = predictions['top_pearson']
plt.scatter(pcorr[:,0],pcorr[:,1],color='k')
plt.scatter(pcorr_top[:,0],pcorr_top[:,1],color='r')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('Input Pearson correlation')
plt.ylabel('Output Pearson correlation')
plt.title('Neuron: ' + NP_neuron)
#%% Assign neuron name
ass_thresh = 1.6
pred_ids = predictions['top_candidates']
pred_met = predictions['top_corr_metric']
keep = pred_met>ass_thresh
assign_ids = pred_ids[keep]
fw.update_class(assign_ids,NP_neuron)


