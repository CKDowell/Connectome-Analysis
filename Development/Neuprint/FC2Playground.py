# -*- coding: utf-8 -*-
"""
Created on Tue May  7 13:23:52 2024

@author: dowel
"""

#%% 
from Stable.SimulationFunctions import sim_functions as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
# from neuprint import Client
# c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
# c.fetch_version()
from neuprint import fetch_neurons,fetch_simple_connections,fetch_synapses,fetch_synapse_connections, NeuronCriteria as NC, SynapseCriteria as SC
#%% Get dorso-ventral locations of FC2 inputs and outputs
ttype = 'PFL3'
neuron_criteria = NC(status='Traced', type=ttype, cropped=False)
syn_criteria = SC(rois='FB', primary_only=True)
#syn_criteria = SC(primary_only=True)
in_syns = fetch_synapses(neuron_criteria, syn_criteria)
da_conns = fetch_synapse_connections(None,neuron_criteria, syn_criteria)
#%% Get pre types
u_FC = da_conns['bodyId_post'].unique()
pre_ns,pre_ns2 = fetch_neurons(da_conns['bodyId_pre'])
utype = pre_ns['type'].unique()
coord_loc = np.zeros((len(utype),3))
syn_num = np.zeros(len(utype))
coords = da_conns[['x_pre','y_pre','z_pre']].to_numpy()
mn_coords = np.min(coords,axis=0)
coords = coords-mn_coords
coords[:,2] = -coords[:,2]
coords[:,2] =coords[:,2]-min(coords[:,2])
for i,u in enumerate(utype):
    uns = pre_ns['bodyId'][pre_ns['type']==u]
    dx = np.isin(da_conns['bodyId_pre'],uns)
    
    tcoords = coords[dx,:]
    med_coords = np.median(tcoords,axis=0)
    syn_num[i] = sum(dx)/len(u_FC)
    if sum(dx)>100:
        coord_loc[i,:] = med_coords
    

dkeep = syn_num>20
utypekeep = utype[dkeep]
coord_loc = coord_loc[dkeep,:]
syn_num_keep = syn_num[dkeep]
plt.close('all')
# Plot hDeltas
plt.figure()
plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1)
hdx = [i  for i,x in enumerate(utypekeep) if 'hDelta' in x]
plt.scatter(coord_loc[hdx,0],coord_loc[hdx,2],s = syn_num_keep[hdx])
for i in hdx:
    plt.annotate(utypekeep[i],(coord_loc[i,0],coord_loc[i,2] ))
plt.title('hDeltas')
plt.xlabel('L-R axis')
plt.ylabel('D-V axis')
# Plot Tangentials
plt.figure()
plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1)
hdx = [i  for i,x in enumerate(utypekeep) if 'FB' in x]
plt.scatter(coord_loc[hdx,0],coord_loc[hdx,2],s = syn_num_keep[hdx])
for i in hdx:
    plt.annotate(utypekeep[i],(coord_loc[i,0],coord_loc[i,2] ))
plt.title('Tangentials')
plt.xlabel('L-R axis')
plt.ylabel('D-V axis')
# Plot PFN
plt.figure()
plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1)
hdx = [i  for i,x in enumerate(utypekeep) if 'PFN' in x]
plt.scatter(coord_loc[hdx,0],coord_loc[hdx,2],s = syn_num_keep[hdx])
for i in hdx:
    plt.annotate(utypekeep[i],(coord_loc[i,0],coord_loc[i,2] ))
plt.title('PFN')
plt.xlabel('L-R axis')
plt.ylabel('D-V axis')
#%% Scatter tangential/helta/PFN synapses
savedir = os.path.join("Y:\\Data\\Connectome\\Connectome Mining\\FC1\\Inputs",ttype)
plt.close('all')
hdx = [i  for i,x in enumerate(utypekeep) if 'hDelta' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_pre'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    #plt.savefig(os.path.join(savedir,savename))
    
    
hdx = [i  for i,x in enumerate(utypekeep) if 'FB' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_pre'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    #plt.savefig(os.path.join(savedir,savename))
    
hdx = [i  for i,x in enumerate(utypekeep) if 'PF' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_pre'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    #plt.savefig(os.path.join(savedir,savename))
    
hdx = [i  for i,x in enumerate(utypekeep) if 'vDelta' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_pre'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    #plt.savefig(os.path.join(savedir,savename))
    
hdx = [i  for i,x in enumerate(utypekeep) if 'FC' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_pre'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    #plt.savefig(os.path.join(savedir,savename))
#%% Get post syns
ttype = 'FC2B'
neuron_criteria = NC(status='Traced', type=ttype, cropped=False)
syn_criteria = SC(rois='FB', primary_only=True)
#syn_criteria = SC(primary_only=True)
in_syns = fetch_synapses(neuron_criteria, syn_criteria)
da_conns = fetch_synapse_connections(neuron_criteria,None, syn_criteria)
#%% Get post types
u_FC = da_conns['bodyId_pre'].unique()
pre_ns,pre_ns2 = fetch_neurons(da_conns['bodyId_post'])
utype = pre_ns['type'].unique()
coord_loc = np.zeros((len(utype),3))
syn_num = np.zeros(len(utype))
coords = da_conns[['x_pre','y_pre','z_pre']].to_numpy()
mn_coords = np.min(coords,axis=0)
coords = coords-mn_coords
coords[:,2] = -coords[:,2]
coords[:,2] =coords[:,2]-min(coords[:,2])
for i,u in enumerate(utype):
    uns = pre_ns['bodyId'][pre_ns['type']==u]
    dx = np.isin(da_conns['bodyId_post'],uns)
    
    tcoords = coords[dx,:]
    med_coords = np.median(tcoords,axis=0)
    syn_num[i] = sum(dx)/len(u_FC)
    if sum(dx)>100:
        coord_loc[i,:] = med_coords
    

dkeep = syn_num>5
utypekeep = utype[dkeep]
coord_loc = coord_loc[dkeep,:]
syn_num_keep = syn_num[dkeep]
plt.close('all')
# Plot hDeltas
plt.figure()
plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1)
hdx = [i  for i,x in enumerate(utypekeep) if 'hDelta' in x]
plt.scatter(coord_loc[hdx,0],coord_loc[hdx,2],s = syn_num_keep[hdx])
for i in hdx:
    plt.annotate(utypekeep[i],(coord_loc[i,0],coord_loc[i,2] ))
plt.title('hDeltas')
plt.xlabel('L-R axis')
plt.ylabel('D-V axis')
# Plot Tangentials
plt.figure()
plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1)
hdx = [i  for i,x in enumerate(utypekeep) if 'FB' in x]
plt.scatter(coord_loc[hdx,0],coord_loc[hdx,2],s = syn_num_keep[hdx])
for i in hdx:
    plt.annotate(utypekeep[i],(coord_loc[i,0],coord_loc[i,2] ))
plt.title('Tangentials')
plt.xlabel('L-R axis')
plt.ylabel('D-V axis')
# Plot PFN
plt.figure()
plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1)
hdx = [i  for i,x in enumerate(utypekeep) if 'PFN' in x]
plt.scatter(coord_loc[hdx,0],coord_loc[hdx,2],s = syn_num_keep[hdx])
for i in hdx:
    plt.annotate(utypekeep[i],(coord_loc[i,0],coord_loc[i,2] ))
plt.title('PFN')
plt.xlabel('L-R axis')
plt.ylabel('D-V axis')
#%%
savedir = os.path.join("Y:\\Data\\Connectome\\Connectome Mining\\FC2\\Outputs",ttype)
plt.close('all')
hdx = [i  for i,x in enumerate(utypekeep) if 'hDelta' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_post'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    plt.savefig(os.path.join(savedir,savename))
    
    
hdx = [i  for i,x in enumerate(utypekeep) if 'FB' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_post'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    plt.savefig(os.path.join(savedir,savename))
    
hdx = [i  for i,x in enumerate(utypekeep) if 'PF' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_post'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    plt.savefig(os.path.join(savedir,savename))
    
hdx = [i  for i,x in enumerate(utypekeep) if 'vDelta' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_post'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    plt.savefig(os.path.join(savedir,savename))
    
hdx = [i  for i,x in enumerate(utypekeep) if 'FC' in x]
for h in hdx:
    
    uns = pre_ns['bodyId'][pre_ns['type']==utypekeep[h]]
    dx = np.isin(da_conns['bodyId_post'],uns)
    tcoords = coords[dx,:]
    plt.figure()
    plt.scatter(coords[:,0],coords[:,2],alpha=0.05,s=1,color='k')
    plt.scatter(tcoords[:,0],tcoords[:,2],s=3,color='r')
    plt.title(utypekeep[h])
    savename = utypekeep[h]+ '.png'
    plt.savefig(os.path.join(savedir,savename))