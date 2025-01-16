# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:51:34 2025

@author: dowel

Tangential PFN shift check

"""

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
#%%
plt.close('all')
ttype = 'PFN.*'
neuron_criteria = NC(status='Traced', type=ttype, cropped=False)
tneurs,tn = fetch_neurons(neuron_criteria)
utypes = tneurs['type'].unique()

outdict = {}
for i,t in enumerate(utypes): 
    upstream_criteria= NC(status='Traced', type='FB.*', cropped=False)
    downstream_criteria = NC(status='Traced', type=t, cropped=False)
    #syn_criteria = SC(rois='FB', primary_only=True)
    conns = fetch_simple_connections(upstream_criteria,downstream_criteria)
    
    upre,ui = np.unique(conns['bodyId_pre'].to_numpy(),return_index=True)
    utypes_pre = conns['type_pre'][ui].to_numpy()
    conside = np.zeros((len(upre),2)) # FB syns, nod syns
    consum = np.zeros((len(upre),2))
    for ui,u in enumerate(upre):
        udx = conns['bodyId_pre']==u
        croi = conns['conn_roiInfo'][udx].to_numpy()
        cell_out = conns['bodyId_post'][udx].to_numpy()
        for ic,c in enumerate(croi):
            tpfn_dx = tneurs['bodyId']==cell_out[ic]
            pfn_neur = tneurs['inputRois'][tpfn_dx].to_numpy()
            L = 'NO(L)' in pfn_neur[0]
            if L:
                sign=-1
               
            else: 
                sign =1 
            tkeys = c.keys()
            if 'NO' in tkeys:
                try:
                    conside[ui,0] += c['NO']['post']*sign
                    consum[ui,0] += c['NO']['post']
                except:
                    conside[ui,0] += c['NO']['pre']*sign
                    consum[ui,0] += c['NO']['pre']
            if 'FB' in tkeys:
                try:
                    conside[ui,1] += c['FB']['post']*sign
                    consum[ui,1] += c['FB']['post']
                except:
                    conside[ui,1] += c['FB']['pre']*sign
                    consum[ui,1] += c['FB']['pre']
                
    plt.figure()
    plt.plot(conside[:,0],color='k')
    plt.plot(conside[:,1],color='r')
    plt.plot(consum[:,0],color='k',linestyle=':')
    plt.plot(consum[:,1],color='r',linestyle=':')
    plt.xticks(np.arange(0,len(conside)),labels=utypes_pre,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.title(t)
    outdict.update({t:{'InIds':upre,'InTypes':utypes_pre,'ConSide':conside}})
#%% Similar analysis with PFL3s        
plt.close('all')
ttype = 'PFL3'

neuron_criteria = NC(status='Traced', type=ttype, cropped=False)
tneurs,tn = fetch_neurons(neuron_criteria)
utypes = tneurs['type'].unique()

outdict = {}
for i,t in enumerate(utypes): 
    upstream_criteria= NC(status='Traced', type='FB.*', cropped=False)
    downstream_criteria = NC(status='Traced', type=t, cropped=False)
    #syn_criteria = SC(rois='FB', primary_only=True)
    conns = fetch_simple_connections(upstream_criteria,downstream_criteria)
    
    upre,ui = np.unique(conns['bodyId_pre'].to_numpy(),return_index=True)
    utypes_pre = conns['type_pre'][ui].to_numpy()
    conside = np.zeros((len(upre),2)) # FB syns, nod syns
    consum = np.zeros((len(upre),2))
    for ui,u in enumerate(upre):
        udx = conns['bodyId_pre']==u
        croi = conns['conn_roiInfo'][udx].to_numpy()
        cell_out = conns['bodyId_post'][udx].to_numpy()
        for ic,c in enumerate(croi):
            #print(ui,ic)
            tpfn_dx = tneurs['bodyId']==cell_out[ic]
            pfn_neur = tneurs['outputRois'][tpfn_dx].to_numpy()
            R = 'RUB(R)' in pfn_neur[0] or 'ROB(R)' in pfn_neur[0] or 'LAL(R)' in pfn_neur[0]
            if R:
                sign=1
               
            else: 
                sign =-1 
            tkeys = c.keys()
            
            if 'FB' in tkeys:
                try:
                    conside[ui,1] += c['FB']['post']*sign
                    consum[ui,1] += c['FB']['post']
                    #print(c['FB']['post'],consum[ui,1])
                except:
                    conside[ui,1] += c['FB']['pre']*sign
                    consum[ui,1] += c['FB']['pre']
                    #print(c['FB']['post'],consum[ui,1])
    plt.figure()
    plt.plot(conside[:,0],color='k')
    plt.plot(conside[:,1],color='r')
    plt.plot(consum[:,0],color='k',linestyle=':')
    plt.plot(consum[:,1],color='r',linestyle=':')
    plt.xticks(np.arange(0,len(conside)),labels=utypes_pre,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.title(t)
    plt.figure()
    plt.plot(conside[:,1]/consum[:,1])
    plt.xticks(np.arange(0,len(conside)),labels=utypes_pre,rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.title(t)
    outdict.update({t:{'InIds':upre,'InTypes':utypes_pre,'ConSide':conside}})    
#%%
for t in utypes:
