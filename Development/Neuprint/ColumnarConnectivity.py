# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:42:08 2025

@author: dowel

Aim of this script is to look at the interconnectivity between select  columnar populations
and see how ring attractor-like the connections are

"""



import Stable.ConnectomeFunctions as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, leaves_list
import os
from neuprint import fetch_neurons, fetch_adjacencies,fetch_synapses, NeuronCriteria as NC
from neuprint import SynapseCriteria as SC
plt.rcParams['pdf.fonttype'] = 42 
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
#%% Tangential neuron analysis
from neuprint import fetch_neurons, NeuronCriteria as NC
from neuprint import queries 

#%% Load data
# To do: extend this to look at all columnar to columnar connections
plt.close('all')
neurons = [
    'hDeltaC',
    'hDeltaF',
    'FC2B',
    # 'hDeltaJ','FC1C','FC2A',
    
    # 'FS1A','FS1B','FS2','FS3','FS4A','FS4B','FS4C',
    # 'hDeltaA','hDeltaB','hDeltaC','hDeltaD','hDeltaE','hDeltaF',
    #'hDeltaG','hDeltaH','hDeltaI','hDeltaJ','hDeltaK','hDeltaL',
           # 'hDeltaM','FC1A','FC1B','FC1C','FC2A','FC2B','FC2C',
           # 'vDeltaA_a','vDeltaA_b','vDeltaB','vDeltaC','vDeltaD','vDeltaE','vDeltaF','vDeltaG','vDeltaH','vDeltaI',
                      #'vDeltaJ','vDeltaK','vDeltaL','vDeltaM'
                      ]
columnar_query = ['PFN','PFR','FS','FC','FR','PFG','PFL','hDelta','vDelta']
prepost_query =       ['pre','pre','post','pre','post','pre','post','pre','pre']
for t_neuron in neurons:
    print(t_neuron)
    prepost = 'pre' # How to order the columns, by pre or post neuron
    df,cf = fetch_adjacencies(NC(type=t_neuron))
    
    ucons1 = df['type'].value_counts().reset_index()
    ucons1.columns = ['type', 'count']
    
    
    ucons = ucons1['type'].to_numpy()
    ucounts = ucons1['count'].to_numpy()
    ucounts = ucounts[ucons!=None]
    ucons = ucons[ucons!=None].astype(str)
    
    con_cols = []
    
    for prefix in columnar_query:
        match =ucons[ np.char.startswith(ucons, prefix)]
        
        con_cols.extend(match)
        
        
    con_cols = np.array(con_cols)
    con_counts  = np.array([])
    ws = cf['weight'].to_numpy()
    prs = cf['bodyId_pre'].to_numpy()
    pst = cf['bodyId_post'].to_numpy()
    cid = bid = df['bodyId'][df['type']==t_neuron].to_numpy()
    for c in con_cols:
        bid = df['bodyId'][df['type']==c].to_numpy()
        dx1 = np.logical_and(np.in1d(prs,bid),np.in1d(pst,cid))
        dx2 = np.logical_and(np.in1d(pst,bid),np.in1d(prs,cid))
        dx = np.logical_or(dx1,dx2)
        con_counts = np.append(con_counts,np.sum(ws[dx]))
    
    
    
    ari = np.argsort(-con_counts)
    
    con_cols = con_cols[ari]
    
    df,cf = fetch_adjacencies(NC(type=con_cols))
    syns = fetch_synapses(NC(type=con_cols,rois=['FB']),SC(rois=['FB']))
    
    con_cols = con_cols[con_cols!=t_neuron]
    
    
     
    
    #Get column order by pre synapse location
    t_ids = df['bodyId'][df['type']==t_neuron].unique()
    t_cols = np.linspace(0,9,len(t_ids),dtype='int').astype(float)+0.01
    synlocs = np.zeros((len(t_ids),3))
    for i,ti in enumerate(t_ids):
        tsyns = syns[syns['bodyId']==ti]
        tsynpre = tsyns[tsyns['type']==prepost]
        synlocs[i,0] = tsynpre['x'].mean()
        synlocs[i,1] = tsynpre['y'].mean()
        synlocs[i,2] = tsynpre['z'].mean()
    
    
    synlocz = (synlocs-np.mean(synlocs,axis=0))/np.std(synlocs,axis=0)
    synlocn = synlocz
    synlocn[:,1] = synlocn[:,1] -np.min(synlocn[:,1] )
    #unfolds fsb into an approximate linear array
    synlocn[synlocn[:,0]<0,1] = -synlocn[synlocn[:,0]<0,1]
    
    reg = LinearRegression().fit(synlocn[:,0][:,np.newaxis],synlocn[:,1])
    vec = np.array([1,reg.coef_[0]])
    vec = vec/np.sqrt(np.sum(vec**2))
    proj = np.matmul(synlocn[:,:2],vec)
    
    cellorder = np.argsort(proj)
    
    t_ids = t_ids[cellorder]
    
    # append t_ids of other neurons
    for i_n,n in enumerate(con_cols):
        print('Col: ',n)
        t_ids2 = df['bodyId'][df['type']==n].unique()
        t_cols = np.append(t_cols,np.linspace(0,9,len(t_ids2),dtype='int').astype(float)+0.02+i_n*0.01)
        synlocs = np.zeros((len(t_ids2),3))
        dx = [ip for ip,prefix in enumerate(columnar_query) if np.char.startswith(n, prefix)]
        prepost = prepost_query[dx[0]]
        for i,ti in enumerate(t_ids2):
            tsyns = syns[syns['bodyId']==ti]
            tsynpre = tsyns[tsyns['type']==prepost]
            synlocs[i,0] = tsynpre['x'].mean()
            synlocs[i,1] = tsynpre['y'].mean()
            synlocs[i,2] = tsynpre['z'].mean()
        
        
        synlocz = (synlocs-np.mean(synlocs,axis=0))/np.std(synlocs,axis=0)
        synlocn = synlocz
        synlocn[:,1] = synlocn[:,1] -np.min(synlocn[:,1] )
        #unfolds fsb into an approximate linear array
        synlocn[synlocn[:,0]<0,1] = -synlocn[synlocn[:,0]<0,1]
        
        reg = LinearRegression().fit(synlocn[:,0][:,np.newaxis],synlocn[:,1])
        vec = np.array([1,reg.coef_[0]])
        vec = vec/np.sqrt(np.sum(vec**2))
        proj = np.matmul(synlocn[:,:2],vec)
        
        cellorder = np.argsort(proj)
        t_ids2 = t_ids2[cellorder]
        t_ids = np.append(t_ids,t_ids2)
    
    
    
    #simpleconmat = np.zeros((len(t_ids),len(t_ids)))
    
    cfc = cf.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
    
    cf_filt = cfc[cfc['bodyId_pre'].isin(t_ids)&cfc['bodyId_post'].isin(t_ids)]
    simpleconmat = (   cf_filt
    .pivot(index='bodyId_pre', columns='bodyId_post', values='weight')
    .reindex(index=t_ids, columns=t_ids, fill_value=0))
    simpleconmat = simpleconmat.to_numpy()
    simpleconmat[np.isnan(simpleconmat)] = 0
    
    # print('Making Matrix')
    # for i,ti in enumerate(t_ids):
    #     print(i,'/',len(t_ids))
    #     tcf = cf[cf['bodyId_pre']==ti]
    #     for i2,ti2 in enumerate(t_ids):
    #         tcf2 = tcf[tcf['bodyId_post']==ti2]
    #         if len(tcf2)>0:
    #             simpleconmat[i,i2] = tcf2['weight'].sum()
    # print('Matrix Made')
    
    print('Making column matrix')
    
    unique_vals, idx = np.unique(t_cols, return_index=True)
    colnum = unique_vals[np.argsort(idx)]
    colmat = np.zeros((len(colnum),len(colnum)))
    for i,ic in enumerate(colnum):
        dx = t_cols==ic
        tmat = np.mean(simpleconmat[dx,:],axis=0)
        for i2,ic2 in enumerate(colnum):
            dx2 = t_cols==ic2
            colmat[i,i2] = np.mean(tmat[dx2])
    
    colmax = np.where(np.diff(colnum)<0)[0]+0.5
    colmax = np.append(-0.5,colmax)
    conames = np.append(t_neuron,con_cols)
    plt.figure()
    # plt.subplot(1,2,1)
    # plt.scatter(synlocs[:,0],synlocs[:,1],c = proj)
    # plt.subplot(1,2,2)
    plt.imshow(colmat,vmax=30)
    plt.xticks(colmax,labels=conames,rotation=90);
    plt.yticks(colmax,labels=conames);
    plt.colorbar()
    plt.title(t_neuron)
    