# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:18:46 2023

@author: dowel
"""

#%% Import packages
from Stable.SimulationFunctions import sim_functions as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
# from neuprint import Client
# c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
# c.fetch_version()
from neuprint import fetch_adjacencies, fetch_neurons, NeuronCriteria as NC

#%% 


#%% Simulate columnar input to FB
S = sf()

columnar_neurons = ['hDeltaA','hDeltaB','hDeltaC','hDeltaD','hDeltaE',
                    'hDeltaF','hDeltaG','hDeltaH','hDeltaI','hDeltaJ',
                    'hDeltaK','hDeltaL','hDeltaM','PFNa','PFNv','PFNd',
                    'PFR','PFNm_a','PFNm_b','PFNp_a','PFNp_b','PFNp_c',
                    'PFNp_d','FC1A','FC1B','FC1C','FC1D',
                    'FC1E','FC2A','FC2B','FC2C','FC3','PFL1','PFL2','PFL3']

FB_network = ['FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*','PFR.*','FR.*','PFG.*','FS.*','PFL.*']
sim_output = {}
for c in columnar_neurons:
    print(c)
    sim_output[c] = S.run_sim_act([c],FB_network,5)

#%% plot effect on hDelta C
savedir = "Y:\Data\Connectome\Connectome Mining\ActivationSimulations\ColumnarStimulation"
layers = [1,2,3] # Look at each iteration layer
cnt = 0
plt.figure(figsize=(10,5))
xticknames = []
this_neuron = 'hDeltaC'
for i in layers:
    for c in columnar_neurons:
        if c == this_neuron:
            continue
        tsim = sim_output[c]
        sim_mat = tsim['MeanActivityType']
        t_names = tsim['TypesSmall']
        n_dx = t_names== this_neuron
        ts = sim_mat[:,n_dx]
        plt.scatter(cnt,ts[i],color='k')
        cnt = cnt+1 
        xticknames = np.append(xticknames,c)
    plt.plot([cnt-0.5,cnt-0.5],[-0.03, 0.03],linestyle='--',color='r')
plt.subplots_adjust(bottom=0.3)
plt.xticks(np.linspace(0,cnt-1,cnt),labels=xticknames,rotation=90)
plt.xlabel('Stimulated neurons')
plt.title('Recipient neuron: ' + this_neuron)
plt.ylabel('Activity')
plt.show()


savepath = os.path.join(savedir,"Col_stim"+this_neuron+".png")
plt.savefig(savepath)

#%% 
S.set_neur_dicts(['PFL3'])
