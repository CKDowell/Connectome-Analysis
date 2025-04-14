# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:39:50 2024

@author: dowel
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
def get_meshes(neuron):

    nprops,ni = fetch_neurons(NC(type=neuron))
   
    savedir = 'Y:\\Data\\Connectome\\Animations'
    sname ='_'.join(neuron)
    savepath = os.path.join(savedir,sname +'.json')
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
get_meshes(['FB2A','FB4L','FB4M','FB5H','FB6H','FB7B'])
