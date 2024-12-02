# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:42:36 2024

@author: dowel

Aim of this script:
    Classify the tangential neurons in the FSB according to
    their connections within the FSB.
    
    This will be done by creating a single matrix where the rows
    are each neuron type and the columns are a 1 - 0 encoding of a connection
    type.
    
    
    This will start simple and only consider inputs/outputs of large strength
    
    Motifs in
    PFN -> tan
    hDelta - > tan
    FC2 -> tan
    FC1 -> tan
    PFR -> tan
    PFG -> tan
    vDelta -> tan
    tan -> tan
    self->tan
    
    Motifs out
    tan ->PFN
    tan ->hDelta axon
    tan ->hDelta dendrite
    tan -> FC2
    tan -> FC1
    tan ->PFR
    tan ->PFG
    tan ->vDelta axon
    tan -> vDelta dendrite
    tan ->PFL3
    tan->self
    
    Reciprocal moniker 
    take combination of the above and say if reciprocal connection
    e.g. hDelta->tan->same hDelta
    tan->tan->tan_same
    

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
out_types, in_types, in_array, out_array, Names = cf.input_output_matrix(['FB.*'])

#%%
synthresh  = 30
# Initialise array

v_names = ['PFN','PFR','PFG','PFL','hDelta','vDelta','FB','FS','FC2','FC1','self']
# For now not considering hDelta axon versus dendrite
motmat = np.zeros((len(Names),len(v_names)*2))
for i,n in enumerate(Names):
    print(n)
    n_df,c_simple = fetch_neurons(NC(type=n))
    n_neur = len(n_df)
    
    tin = in_array[i,:]/n_neur
    
    tout = out_array[i,:]/n_neur
    
    tinp = tin/np.sum(tin)
    toutp = tout/np.sum(tout)
    
    tin_names = in_types[tin>synthresh]
    tout_names = out_types[tout>synthresh]
    v_names2 = v_names
    v_names2[-1] = n
    for iv,v in enumerate(v_names2):
        it = [1 for t in tin_names if v in t]
        if sum(it)>0:
            motmat[i,iv] = 1 
            
        it = [1 for t in tout_names if v in t]
        if sum(it)>0:
            motmat[i,iv+len(v_names)] = 1



#%%
plt.close('all')
plt.imshow(motmat,aspect='auto',interpolation='none',cmap='Greys_r')

tick_names = ['in_PFN','in_PFR','in_PFG','in_PFL','in_hDelta','in_vDelta','in_FB','in_FS','in_FC2','in_FC1','in_self',
              'out_PFN','out_PFR','out_PFG','out_PFL','out_hDelta','out_vDelta','out_FB','out_FS','out_FC2','out_FC1','out_self'
              
              ]
plt.xticks(np.arange(0,len(tick_names)),labels=tick_names,rotation=90)
plt.yticks(np.arange(0,len(Names)),labels=Names,fontsize=8)

um = np.unique(motmat,axis=0)
uzero = np.where(np.sum(um,axis=1)==0)[0]
udx = uzero
for i,u in enumerate(um):
    if np.sum(u)==0:
        continue
    
    umn = motmat-u
    dx = np.where(np.sum(np.abs(umn),axis=1)==0)
    udx = np.append(dx,udx)
    
plt.figure()
plt.imshow(motmat[udx,:],aspect='auto',interpolation='none',cmap='Blues')
    
tick_names = ['in_PFN','in_PFR','in_PFG','in_PFL','in_hDelta','in_vDelta','in_FB','in_FS','in_FC2','in_FC1','in_self',
              'out_PFN','out_PFR','out_PFG','out_PFL','out_hDelta','out_vDelta','out_FB','out_FS','out_FC2','out_FC1','out_self'
              
              ]
plt.xticks(np.arange(0,len(tick_names)),labels=tick_names,rotation=90)
plt.yticks(np.arange(0,len(Names[udx])),labels=Names[udx],fontsize=8)


motmat2 = motmat[np.sum(motmat,axis=1)>0,:]
nnames = Names[np.sum(motmat,axis=1)>0]
cluster_in,dmat_in = cf.hier_cosine(motmat2,0.7)
z_in = cf.linkage_order(cluster_in)
plt.figure()
plt.imshow(motmat2[z_in,:],aspect='auto',interpolation='none',cmap='Blues')
plt.xticks(np.arange(0,len(tick_names)),labels=tick_names,rotation=90)
plt.yticks(np.arange(0,len(nnames[z_in])),labels=nnames[z_in],fontsize=8)