# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:53:12 2024

@author: dowel

The script will plot some connectivity figures for hDeltaA and FB4Z

This will include:
    1. A connectivity matrix of the two neurons showing reciprocity
    2. 


"""

import Stable.ConnectomeFunctions as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, leaves_list
import os
plt.rcParams['pdf.fonttype'] = 42 
#%% Tangential neuron analysis
from neuprint import fetch_neurons, NeuronCriteria as NC
from neuprint import queries 
#%%
nprops,ni = fetch_neurons(NC(type=['FB4Z','hDeltaA']))
#%% Construct json
n_name = 'FB4Z_hDeltaA'
savedir = 'Y:\\Data\\Connectome\\Animations'
savepath = os.path.join(savedir,n_name+'.json')
import json
ndict = {'neurons': {'source': "https://hemibrain-dvid.janelia.org/api/node/31597/segmentation_meshes"
                     },"animation": [
    ["frameCamera", {"bound": "neurons"}],
    ["orbitCamera", {"duration": 10}]]}
nids = nprops['bodyId']
for i, n in enumerate(nids):
    ndict['neurons'].update({'n'+str(i+1): [n]})
    
    
    
    
with open(savepath, 'w') as outfile:
    json.dump(ndict, outfile)
#%% 
out_dict  = cf.defined_in_out(['FB.*','hDelta.*','FC.*','PF.*','vDelta.*'],['FB.*','hDelta.*','FC.*','PF.*','vDelta.*'])
#%% 
# Find neurons whose top inputs = top outputs
Names = out_dict['in_types']
conmat = out_dict['con_mat_sum']
for i,n in enumerate(Names):
    ins = conmat[:,i]
    ins = ins/np.sum(ins)
    outs = conmat[i,:]
    outs = outs/np.sum(outs)
    #indx = np.argmax(ins)
    #outdx = np.argmax(outs)
    
    touts = Names[outs>0.2]
    tins = Names[ins>0.2]
    shrd = [t for i,t in enumerate(tins) if t in touts]
    if np.shape(shrd)[0]>0:
        print(n,' ',shrd)
        ndx = Names==shrd
        print(outs[ndx]*100,ins[ndx]*100)
    
#%%
plt.plot(outs)
plt.xticks(np.arange(0,len(outs)),labels=Names,rotation=90,fontsize=6)
plt.plot(ins)