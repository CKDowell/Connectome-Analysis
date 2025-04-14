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
out_types, in_types, in_array, out_array, Names,tcounts = cf.input_output_matrix(['FB.*'])
in_array[np.isnan(in_array)] = 0
out_array = in_array/tcounts[:,np.newaxis]
in_array = in_array/tcounts[:,np.newaxis]
#%%
synthresh  = 30
# Initialise array

v_names = ['PFN','PFR','PFG','PFL','hDelta','vDelta','FB','FS','FR','FC2','FC1','self']
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
#%% 
from Stable.BasicNeuronProperties import neuron_properties as NP
npr = NP()
mbondx = [i for i,e in enumerate(in_types) if 'MBON' in e]
MBON_names = in_types[mbondx]
MBON_in  = in_array[:,mbondx]
mbon_in_sum = np.sum(MBON_in,axis=1)
i = np.argsort(-mbon_in_sum)
num_neur = 20
for m in range(num_neur):
    tmn = MBON_in[i[m],:]
    tdx = tmn>0
    tmnames = MBON_names[tdx]
    print(tmnames)
    tmnw =tmn[tdx]
    IT = np.argsort(-tmnw)
    tmnw = tmnw[IT]
    tmnames = tmnames[IT]
    xoff= 0
    yoff=0
    for it,t in enumerate(tmnw):
        yoff = yoff+0.075
        val = npr.MBON_valence_query(tmnames[it])[0]
        if val==-1:
            colour = np.array([235,0,255])/255
        elif val==1:
            colour= np.array([63,236,158])/255
        plt.fill_between([xoff,t+xoff],[-m+0.25-yoff,-m+0.25-yoff],[-m+0.15-yoff,-m+0.15-yoff],color=colour)
        if t>20:
            plt.text(np.mean([xoff,t+xoff]),-m+0.4-yoff,tmnames[it],fontsize=8,horizontalalignment='center')
        
        #plt.fill_between([xoff,t+xoff],[-m+0.25,-m+0.25],[-m-0.25,-m-0.25],color=colour)
       #plt.plot([xoff+0.5,xoff+0.5],[-m+0.25,-m-0.25],color='k')
        xoff = t+xoff
plt.yticks(-np.arange(0,num_neur),labels=Names[i[:num_neur]])
plt.xlabel('Mean synapse count')
savedir = r'Y:\Presentations\2025\02_BSV\MB'
plt.savefig(os.path.join(savedir,'MBON_tan.pdf'))
