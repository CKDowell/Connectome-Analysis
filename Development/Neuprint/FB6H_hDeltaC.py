# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:24:57 2025

@author: dowel

Code will assess synapse locations from FB6H onto hDeltaC

"""
from neuprint import Client
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
import pandas as pd
import numpy as np
from neuprint import fetch_neurons,fetch_simple_connections,fetch_synapses,fetch_synapse_connections, NeuronCriteria as NC, SynapseCriteria as SC
import matplotlib.pyplot as plt
import pickle as pkl
import os
plt.rcParams['pdf.fonttype'] = 42 

#%%
da_criteria = SC(rois='FB', primary_only=True)

neuron_df = fetch_synapse_connections(NC(type='FB6H'),NC(type='hDeltaC'),da_criteria)
#%%
pre_synloc =  neuron_df[['x_pre','y_pre','z_pre']].to_numpy()
post_synloc =  neuron_df[['x_post','y_post','z_post']].to_numpy()
plt.scatter(pre_synloc[:,0],pre_synloc[:,1],s=5,color='m')
plt.scatter(post_synloc[:,0],post_synloc[:,1],s=5,color='b')
#%%
h_Delta_out_syn = fetch_synapse_connections(NC(type='hDeltaC'),None,da_criteria)
h_Delta_in_syn = fetch_synapse_connections(None,NC(type='hDeltaC'),da_criteria)
#%% 
plt.close('all')
savedir = r'Y:\Data\Connectome\Connectome Mining\FB6H'
xyz_out = h_Delta_out_syn[['x_pre','y_pre','z_pre']].to_numpy()
xyz_in = h_Delta_in_syn[['x_post','y_post','z_post']].to_numpy()


out_ROI = h_Delta_out_syn['bodyId_post'].to_numpy()
uROI = np.unique(out_ROI)
uNeurons,_ = fetch_neurons(uROI)
uROI = uNeurons['bodyId'].to_numpy()
u_types = uNeurons['type'].to_numpy()
roi_thresh = 1/0.008# 1um assuming 8nm resolution
u_utypes = np.unique(u_types[(u_types!=None)])
typecount =np.zeros_like(u_utypes)

in_ROI = h_Delta_in_syn['bodyId_pre'].to_numpy()
iuROI = np.unique(in_ROI)
iuNeurons,_ = fetch_neurons(iuROI)
iuROI = iuNeurons['bodyId'].to_numpy()
iu_types = iuNeurons['type'].to_numpy()
iu_utypes = np.unique(iu_types[(iu_types!=None)])
iu_utypes = iu_utypes[iu_utypes!='FB6H'] # remove FB6H
itypecount =np.zeros_like(iu_utypes)

for i,xyz in enumerate(post_synloc):
    df = xyz_out-xyz
    dfmod = np.sum(df**2,axis=1)**0.5
    close = dfmod<roi_thresh
    if np.sum(close)>0:
        tR = out_ROI[close]
        tdx = np.in1d(uROI,tR)
        ttype = u_types[tdx]
        ttdx = np.in1d(u_utypes,ttype)
        typecount[ttdx] += 1
        print(ttype)
        
    df = xyz_in-xyz
    dfmod = np.sum(df**2,axis=1)**0.5
    close = dfmod<roi_thresh
    if np.sum(close)>0:
        tR = in_ROI[close]
        tdx = np.in1d(iuROI,tR)
        ttype = iu_types[tdx]
        ttdx = np.in1d(iu_utypes,ttype)
        itypecount[ttdx] += 1
        print(ttype)
    

# Need to compare to random sampling of synapses
minplot = 10
u_utypesplot = u_utypes[typecount>minplot]
typecount_plot = typecount[typecount>minplot]
tcr = np.argsort(-typecount_plot)
yplt = np.zeros((len(typecount_plot),2))
yplt[:,0] = typecount_plot[tcr]/len(post_synloc)
xplt = np.zeros_like(yplt)
xplt[:,0] = np.arange(0,len(typecount_plot))
xplt[:,1] =np.arange(0,len(typecount_plot))
plt.figure()
plt.plot(xplt.T,yplt.T,color='k')
plt.xticks(np.arange(0,len(typecount_plot)),labels=u_utypesplot[tcr],rotation=90)
plt.subplots_adjust(bottom=0.2)
plt.ylabel('Proportion of FB6H-hDeltaC synapses')
plt.title('Within 1 um post-synaptic to hDeltaC')
plt.savefig(os.path.join(savedir,'hDeltaC_FB6H_downstreamClose.png'))
plt.savefig(os.path.join(savedir,'hDeltaC_FB6H_downstreamClose.pdf'))

iu_utypesplot = iu_utypes[itypecount>minplot]
itypecount_plot = itypecount[itypecount>minplot]
tcr = np.argsort(-itypecount_plot)
yplt = np.zeros((len(itypecount_plot),2))
yplt[:,0] = itypecount_plot[tcr]/len(post_synloc)
xplt = np.zeros_like(yplt)
xplt[:,0] = np.arange(0,len(itypecount_plot))
xplt[:,1] =np.arange(0,len(itypecount_plot))
plt.figure()
plt.plot(xplt.T,yplt.T,color='k')
plt.xticks(np.arange(0,len(itypecount_plot)),labels=iu_utypesplot[tcr],rotation=90)
plt.subplots_adjust(bottom=0.2)
plt.ylabel('Proportion of FB6H-hDeltaC synapses')
plt.title('Within 1 um pre-synaptic to hDeltaC')
plt.savefig(os.path.join(savedir,'hDeltaC_FB6H_upstreamClose.png'))
plt.savefig(os.path.join(savedir,'hDeltaC_FB6H_upstreamClose.pdf'))

#%% Same for FB4M

da_criteria = SC(rois='FB', primary_only=True)

neuron_df = fetch_synapse_connections(NC(type='FB4M'),NC(type='hDeltaC'),da_criteria)

pre_synloc =  neuron_df[['x_pre','y_pre','z_pre']].to_numpy()
post_synloc =  neuron_df[['x_post','y_post','z_post']].to_numpy()
plt.scatter(pre_synloc[:,0],pre_synloc[:,1],s=5,color='m')
plt.scatter(post_synloc[:,0],post_synloc[:,1],s=5,color='b')

h_Delta_out_syn = fetch_synapse_connections(NC(type='hDeltaC'),None,da_criteria)
h_Delta_in_syn = fetch_synapse_connections(None,NC(type='hDeltaC'),da_criteria)
#
xyz_out = h_Delta_out_syn[['x_pre','y_pre','z_pre']].to_numpy()
xyz_in = h_Delta_in_syn[['x_post','y_post','z_post']].to_numpy()


out_ROI = h_Delta_out_syn['bodyId_post'].to_numpy()
uROI = np.unique(out_ROI)
uNeurons,_ = fetch_neurons(uROI)
uROI = uNeurons['bodyId'].to_numpy()
u_types = uNeurons['type'].to_numpy()
roi_thresh = 0.5/0.008# .25 um assuming 8nm resolution
u_utypes = np.unique(u_types[(u_types!=None)])
typecount =np.zeros_like(u_utypes)

in_ROI = h_Delta_in_syn['bodyId_pre'].to_numpy()
iuROI = np.unique(in_ROI)
iuNeurons,_ = fetch_neurons(iuROI)
iuROI = iuNeurons['bodyId'].to_numpy()
iu_types = iuNeurons['type'].to_numpy()
iu_utypes = np.unique(iu_types[(iu_types!=None)])
iu_utypes = iu_utypes[iu_utypes!='FB4M'] # remove FB6H
itypecount =np.zeros_like(iu_utypes)

for i,xyz in enumerate(post_synloc):
    df = xyz_out-xyz
    dfmod = np.sum(df**2,axis=1)**0.5
    close = dfmod<roi_thresh
    if np.sum(close)>0:
        tR = out_ROI[close]
        tdx = np.in1d(uROI,tR)
        ttype = u_types[tdx]
        ttdx = np.in1d(u_utypes,ttype)
        typecount[ttdx] += 1
        print(ttype)
        
    df = xyz_in-xyz
    dfmod = np.sum(df**2,axis=1)**0.5
    close = dfmod<roi_thresh
    if np.sum(close)>0:
        tR = in_ROI[close]
        tdx = np.in1d(iuROI,tR)
        ttype = iu_types[tdx]
        ttdx = np.in1d(iu_utypes,ttype)
        itypecount[ttdx] += 1
        print(ttype)
    

# Need to compare to random sampling of synapses
u_utypesplot = u_utypes[typecount>0]
typecount_plot = typecount[typecount>0]
tcr = np.argsort(-typecount_plot)
yplt = np.zeros((len(typecount_plot),2))
yplt[:,0] = typecount_plot[tcr]
xplt = np.zeros_like(yplt)
xplt[:,0] = np.arange(0,len(typecount_plot))
xplt[:,1] =np.arange(0,len(typecount_plot))
plt.figure()
plt.plot(xplt.T,yplt.T,color='k')
plt.xticks(np.arange(0,len(typecount_plot)),labels=u_utypesplot[tcr],rotation=90)

iu_utypesplot = iu_utypes[itypecount>0]
itypecount_plot = itypecount[itypecount>0]
tcr = np.argsort(-itypecount_plot)
yplt = np.zeros((len(itypecount_plot),2))
yplt[:,0] = itypecount_plot[tcr]
xplt = np.zeros_like(yplt)
xplt[:,0] = np.arange(0,len(itypecount_plot))
xplt[:,1] =np.arange(0,len(itypecount_plot))
plt.figure()
plt.plot(xplt.T,yplt.T,color='k')
plt.xticks(np.arange(0,len(itypecount_plot)),labels=iu_utypesplot[tcr],rotation=90)