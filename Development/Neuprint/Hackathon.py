# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:34:02 2024

@author: dowel


Script created for the hackathon on 19th July 2024

Task identify interesting motifs:
    1. FB4Z/hDeltaA 
    2. Large FB neuron numbers - how similar to ring neuron
    3. 


"""

import Stable.ConnectomeFunctions as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, leaves_list
import os
from neuprint import fetch_neurons, fetch_adjacencies, NeuronCriteria as NC

plt.rcParams['pdf.fonttype'] = 42 
#%% Tangential neuron analysis
from neuprint import fetch_neurons, NeuronCriteria as NC
from neuprint import queries 

#%% 

df,cf = fetch_adjacencies(NC(type='PFR_a'),NC(type='FB4P_b'))
#%%
pbdx = df['type']=='FB4P_b'

pdids =df['bodyId'][pbdx].to_numpy()
pfrids = df['bodyId'][~pbdx].to_numpy()
plotmat = np.zeros((len(pfrids),len(pdids)))
columns = np.empty(len(pfrids))
pfrinst = df['instance'][~pbdx].to_numpy()
for i,p in enumerate(pfrinst):
    columns[i] = int(p.split('C')[1][0])
   
csort = np.argsort(columns)
pfrids = pfrids[csort]
for i,ids in enumerate(pfrids):
    condx = cf['bodyId_pre']==ids
    t_p = cf['bodyId_post'][condx].to_numpy()
    tw= cf['weight'][condx].to_numpy()
    for it,tp in enumerate(t_p):
        pdx = pdids==tp
        plotmat[i,pdx] = tw[it]
    
plt.imshow(plotmat,vmin=0,vmax=20)
plt.yticks(np.arange(0,len(columns)),labels=pfrinst[csort])
plt.figure()
plt.plot(plotmat)
#%% Defined in out
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
#%% Full FSB connectivity matrix
out_dictF = cf.defined_in_out_full(['FB.*','hDelta.*','FC.*','PF.*','vDelta.*'],['FB.*','hDelta.*','FC.*','PF.*','vDelta.*'])
#%% 

out_dictF = cf.defined_in_out_full(['.*'],['.*'])

#%%
plt.close('all')
cm = out_dictF['conmat']
ndf = out_dictF['nDF']
types = ndf['type'].to_numpy()
t_types = ['FB4R','FB4X','FB5AB','FB4P_b','FB5I','FB4Z','FB4M','FB4L',
           'hDeltaA','hDeltaB','hDeltaC','hDeltaD','hDeltaE','hDeltaF','hDeltaG',
           'hDeltaH','hDeltaI','hDeltaJ','hDeltaK']
t_types = ['hDeltaA','hDeltaB','hDeltaC','hDeltaD','hDeltaE','hDeltaF','hDeltaG',
           'hDeltaH','hDeltaI','hDeltaJ','hDeltaK']
plt.figure()

for i,t in enumerate(t_types):

    tdx = types==t
    #plt.imshow(cm[:,tdx],vmin=-10,vmax=10,cmap='coolwarm')
    x = cm[:,tdx]
    y = cm[tdx,:].transpose()
    x = x/np.sum(x,axis=0)
    y = y/np.sum(y,axis=0)
    gz = np.logical_or(x>0.01,y>0.01)
    gz2 = np.sum(gz,axis=1)>0
    x = x[gz2,:]
    y = y[gz2,:]
    plt.subplot(3,4,i+1)
    #plt.figure()
    t_types = types[gz2]
    utypes = np.unique(t_types)
    cmap = plt.cm.hsv(np.linspace(0, 1, len(utypes)))
    cvec = np.zeros_like(x)
    for i,ut in enumerate(utypes):
        ut_dx = t_types==ut
        cvec[ut_dx,:] = i
        
        
    plt.scatter(x[:],y[:],c =cvec)
    for i,ut in enumerate(utypes):
        if 'FB' in ut:
            ut_dx = t_types==ut
            xmn = np.mean(x[ut_dx])
            ymn = np.mean(y[ut_dx])
            plt.text(xmn,ymn,ut,color=[0.8,0.3,0.3],fontsize=20)
    plt.title(t)
    mx = np.max(x)
    my = np.max(y)
    mxy = max(mx,my)
    plt.xlim([0,mxy])
    plt.ylim([0,mxy])
    plt.xlabel('Post-neuron')
    plt.ylabel('Pre-neuron')
#plt.plot(cm[:,tdx],color='k')
#plt.plot(-cm[tdx,:].transpose(),color='r')
#%%
def save_top_in_out(neuron,savedir,threshold=10):
    ndf,ndf2 = fetch_neurons(NC(type=neuron))
    typelib_u, t_inputs, ncells = cf.top_inputs(neuron)
    typelib_ou, t_outputs, ncells_o =cf.top_outputs(neuron)
    
    
    plt.figure(figsize=(4,10))
    ndx = typelib_ou!='None'
    out_mean = t_outputs[ndx]/len(ndf)
    pltdx = out_mean>threshold
    ploty = np.zeros((2,sum(pltdx)))
    py = out_mean[pltdx]
    I = np.argsort(py)
    ploty[0,:] = py[I]
    plotlabs = typelib_ou[ndx]
    plotlabs = plotlabs[pltdx]
    x = np.zeros((2,sum(pltdx)))
    x[0,:] = np.arange(0,sum(pltdx))
    x[1,:] = np.arange(0,sum(pltdx))
    plt.plot(ploty,x,linewidth=8,color=[0.8,0.8,0.8])
    plt.yticks(x[0,:],labels=plotlabs[I])
    plt.xlabel('Mean synapse count')
    plt.subplots_adjust(left=0.3)
    plt.ylim([np.min(x[:])-0.5,np.max(x[:])+0.5])
    plt.title('Top ' + neuron + ' outputs')
    savename = os.path.join(savedir,'Top' +neuron+'Outputs.pdf')
    plt.savefig(savename)
    savename = os.path.join(savedir,'Top' +neuron+'Outputs.png')
    plt.savefig(savename)
    
    plt.figure(figsize=(4,10))
    ndx = typelib_u!='None'
    out_mean = t_inputs[ndx]/len(ndf)
    pltdx = out_mean>threshold
    ploty = np.zeros((2,sum(pltdx)))
    py = out_mean[pltdx]
    I = np.argsort(py)
    ploty[0,:] = py[I]
    plotlabs = typelib_u[ndx]
    plotlabs = plotlabs[pltdx]
    x = np.zeros((2,sum(pltdx)))
    x[0,:] = np.arange(0,sum(pltdx))
    x[1,:] = np.arange(0,sum(pltdx))
    plt.plot(ploty,x,linewidth=8,color=[0.8,0.8,0.8])
    plt.yticks(x[0,:],labels=plotlabs[I])
    plt.xlabel('Mean synapse count')
    plt.subplots_adjust(left=0.3)
    plt.ylim([np.min(x[:])-0.5,np.max(x[:])+0.5])
    plt.title('Top ' + neuron + ' inputs')
    savename = os.path.join(savedir,'Top' +neuron+'Inputs.pdf')
    plt.savefig(savename)
    savename = os.path.join(savedir,'Top' +neuron+'Inputs.png')
    plt.savefig(savename)
    
    
#%%
plt.close('all')
savedir = "Y:\\Presentations\\2024\\November\\Panels"
neurons =['FC2B','FC2A','FC2C','FB4R','PFGs','PFR_a','FB4X','FB4P_b','FB5I','FB4M','FB5A']
for n in neurons:
    save_top_in_out(n,savedir)
    
 


#%%


