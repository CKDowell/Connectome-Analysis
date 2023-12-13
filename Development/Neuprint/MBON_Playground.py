# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 09:19:39 2023

@author: dowel
"""

#%% Import packages
from Stable.SimulationFunctions import sim_functions as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
from Stable.ConnectomeFunctions import  defined_in_out, hier_cosine, linkage_order
from Stable.BasicNeuronProperties import neuron_properties as neurP
# from neuprint import Client
# c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
# c.fetch_version()
from neuprint import fetch_adjacencies, fetch_neurons, NeuronCriteria as NC
#%% Get connectivity matrix to tangential neurons

con_dict = defined_in_out('MBON.*','FB.*')
mb_prop = neurP
mb_prop.MBON_compartment_valence(mb_prop)

# %% luster signed con matrix
con_mat = con_dict['con_mat_sign']
cmsum = np.sum(abs(con_mat),axis=1)
zs = cmsum == 0

con_mat = con_mat[~zs,:]
zx = np.sum(con_mat,axis=0)==0
con_mat = con_mat[:,~zx]

cluster,dmat = hier_cosine(np.transpose(con_mat),1)
z = linkage_order(cluster)
plt.figure(figsize=(20,10))
ax = plt.subplot()
ynames = con_dict['in_types'][~zs]
ydx = np.in1d(mb_prop.MBON_dict['MBONs'],ynames)
val = mb_prop.MBON_dict['Valence'][ydx]
comps = mb_prop.MBON_dict['Compartments_cat'][ydx]

I = np.argsort(val)
ynames= ynames[I]
comps = comps[I]
con_plot = con_mat[:,z]
con_plot = con_plot[I,:]
plt.imshow(con_plot,vmin=-50,vmax = 50,interpolation=None,aspect='auto',cmap= 'coolwarm')



xnames = con_dict['out_types'][~zx]
xnames = xnames[z]
yt = np.linspace(0,len(ynames)-1,len(ynames))
xt = np.linspace(0,len(xnames)-1,len(xnames))

cmap = np.array([[39,100,25],
                 [0, 0, 0],
                 [142,1,82],
])/255



ynames_new = ynames.copy()
for i,y in enumerate(ynames):
    ynames_new[i] = comps[i] + ': '+ y 

plt.yticks(yt,labels=ynames_new)


for i, ytick in enumerate(ax.get_yticklabels()):
    i
    tmb = ynames[i]
    dx = mb_prop.MBON_dict['MBONs']==tmb
    val = mb_prop.MBON_dict['Valence'][dx]+1
    
    ytick.set_color(cmap[val,:])
    
plt.ylabel('Pre-synaptic neurons')
plt.xlabel('Post-synaptic neurons')
plt.xticks(xt,labels=xnames,rotation = 90)
plt.subplots_adjust(bottom=0.2)
plt.colorbar(shrink=0.5,fraction=0.1,ticks = [-100,-50,-25,0,25,50,100])

plt.show()
savedir = "Y:\Data\Connectome\Connectome Mining\MBON_outputs"
savefile = os.path.join(savedir,'MBON_FB.png')
plt.savefig(savefile)
plt.rcParams['pdf.fonttype'] = 42 
savedir = "Y:\Data\Connectome\Connectome Mining\MBON_outputs"
savefile = os.path.join(savedir,'MBON_FB.pdf')
plt.savefig(savefile)
#%% Get con matrix for Tangential outputs
con_dict2 = defined_in_out(xnames,xnames)
#%%
z_order = [np.where(con_dict2['in_types']==t)[0][0] for i,t in enumerate(xnames) ]

plt.figure(figsize=(18,18))
con_mat = con_dict2['con_mat']
con_mat = con_mat[z_order,:]
con_mat = con_mat[:,z_order]
plt.imshow(con_mat,vmin=-100,vmax = 100,interpolation=None,cmap= 'coolwarm')
xt = np.linspace(0,len(xnames)-1,len(xnames))
plt.ylabel('Pre-synaptic neurons')
plt.xlabel('Post-synaptic neurons')
plt.xticks(xt,labels=xnames,rotation = 90,fontsize=8)
plt.yticks(xt,labels=xnames,fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.show()
savedir = "Y:\Data\Connectome\Connectome Mining\MBON_outputs"
savefile = os.path.join(savedir,'MBON_FB_interconnections.png')
plt.savefig(savefile)
plt.rcParams['pdf.fonttype'] = 42 
savefile = os.path.join(savedir,'MBON_FB_interconnections.pdf')
plt.savefig(savefile)
#%% Get con matrix for Tangential outputs to columnar outputs
con_dict2 = defined_in_out(xnames,['PFL.*','FR.*','FC.*','hDelta.*',])
#%%
con_mat = con_dict2['con_mat']
cluster,dmat = hier_cosine(np.transpose(con_mat),1)
z = linkage_order(cluster)
#%%
plt.figure(figsize=(8,18))
z_order = [np.where(con_dict2['in_types']==t)[0][0] for i,t in enumerate(xnames) ]
con_mat = con_dict2['con_mat']
con_mat = con_mat[z_order,:]
ynames = con_dict2['out_types']
plt.imshow(con_mat[:,z],vmin=-100,vmax = 100,interpolation=None,aspect='auto',cmap= 'coolwarm')
yt = np.linspace(0,len(xnames)-1,len(xnames))
xt = np.linspace(0,len(ynames)-1,len(ynames))
plt.ylabel('Pre-synaptic neurons')
plt.xlabel('Post-synaptic neurons')
plt.xticks(xt,labels=ynames[z],rotation = 90,fontsize=8)
plt.yticks(yt,labels=xnames,fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.show()

savedir = "Y:\Data\Connectome\Connectome Mining\MBON_outputs"
savefile = os.path.join(savedir,'MBON_FB_to_output.png')
plt.savefig(savefile)
plt.rcParams['pdf.fonttype'] = 42 
savefile = os.path.join(savedir,'MBON_FB_to_output.pdf')
plt.savefig(savefile)

#%% Explore model motif
con_dict2 = defined_in_out(['FB5I','FB5AB','FB6P','FB5J'],['FB5I','FB5AB','FB6P','FB5J','hDeltaC'])
#%%
con_mat = con_dict2['con_mat'].copy()
plt.imshow(con_mat,vmin=-50,vmax = 50,interpolation=None,aspect='auto',cmap= 'coolwarm')
ynames = con_dict2['in_types']
xnames = con_dict2['out_types']
xt = np.linspace(0,len(xnames)-1,len(xnames))
yt = np.linspace(0,len(ynames)-1,len(ynames))
plt.xticks(xt,labels=xnames,rotation = 90,fontsize=8)
plt.yticks(yt,labels=ynames,fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.ylabel('Pre-synaptic neurons')
plt.xlabel('Post-synaptic neurons')
plt.show()
savedir = "Y:\Data\Connectome\Connectome Mining\MBON_outputs"
savefile = os.path.join(savedir,'FB5I_hDeltacC_network.png')
plt.savefig(savefile)
plt.rcParams['pdf.fonttype'] = 42 
savefile = os.path.join(savedir,'FB5I_hDeltacC_network.pdf')
plt.savefig(savefile)
#%%
con_mat = con_dict2['con_mat'].copy()
con_mat[[1, 2],:] = -con_mat[[1, 2],:]
plt.imshow(con_mat,vmin=-50,vmax = 50,interpolation=None,aspect='auto',cmap= 'coolwarm')
ynames = con_dict2['in_types']
xnames = con_dict2['out_types']
xt = np.linspace(0,len(xnames)-1,len(xnames))
yt = np.linspace(0,len(ynames)-1,len(ynames))
plt.xticks(xt,labels=xnames,rotation = 90,fontsize=8)
plt.yticks(yt,labels=ynames,fontsize=8)
plt.subplots_adjust(bottom=0.2)
plt.ylabel('Pre-synaptic neurons')
plt.xlabel('Post-synaptic neurons')
plt.show()
savedir = "Y:\Data\Connectome\Connectome Mining\MBON_outputs"
savefile = os.path.join(savedir,'FB5I_hDeltacC_network_weight.png')
plt.savefig(savefile)
plt.rcParams['pdf.fonttype'] = 42 
savefile = os.path.join(savedir,'FB5I_hDeltacC_network_weight.pdf')
plt.savefig(savefile)