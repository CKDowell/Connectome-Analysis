# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:36:15 2024

@author: dowel
"""

# Idea of code is to get dopamine neuron synapse locations
# Then get adjacent synapses
# See what the organisation looks like
# Catalogue motifs and see if they match hypotheses I have
#%%
from neuprint import Client
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
import pandas as pd
import numpy as np
from neuprint import fetch_neurons,fetch_simple_connections,fetch_synapses,fetch_synapse_connections, NeuronCriteria as NC, SynapseCriteria as SC
import matplotlib.pyplot as plt
import pickle as pkl
import os
#%%
da_neurons = ['FB4R','FB2A','FB4M','FB4L','FB5H','FB6H','FB7B']
pltdir = 'Y:\Presentations\\2024\\MarchLabMeeting\\Figures'
plt.close('all')
for dn in da_neurons:
    neuron_criteria = NC(status='Traced', type=dn, cropped=False)
    da_criteria = SC(rois='FB', type='pre', primary_only=True)
    da_syns = fetch_synapses(neuron_criteria, da_criteria)
    
    da_conns = fetch_synapse_connections(neuron_criteria, None, da_criteria)
    
    n_id = da_syns['bodyId'].to_numpy()
    nu = np.unique(n_id)
    #plt.scatter(da_syns['x'][n_id==nu[1]],da_syns['y'][n_id==nu[1]],s = 10)
    
    
    # Get index of post_types
    post_neurons, _ = fetch_neurons(da_conns['bodyId_post'].unique())
    u_t = post_neurons['type'].unique()
    
    # 
    pair_mat = np.zeros((len(u_t),len(u_t)))
    pre_synloc =  da_syns[['x','y','z']].to_numpy()
    pre_syn_long = da_conns[['x_pre','y_pre','z_pre']]
    for i,p in enumerate(pre_synloc):
        df =pre_syn_long-p
        df_mag = np.sqrt(np.sum(df**2,axis=1))
        dx = df_mag==0 
        post_id = da_conns['bodyId_post'][dx]
        pdx = np.isin(post_neurons['bodyId'],post_id)
        ts = post_neurons['type'][pdx].to_numpy()
        ts = ts[ts!=None]
        u_ts = np.unique( ts)
        for u in u_ts:
            t_u = np.where(u_t==u)[0]
            tdx = ts[ts!=u]
            for t in tdx:
                t_l = np.where(u_t==t)[0]
                pair_mat[t_u,t_l] = pair_mat[t_u,t_l]+1
                
            
            
        print(np.sum(df_mag==0))
    
    pltmat = pair_mat
    xdx = np.max(pair_mat,axis=1)>20
    pltmat = pltmat[xdx,:]
    pltmat = pltmat[:,xdx]
    plt.figure()
    plt.imshow(pltmat,aspect='auto',interpolation='none',vmin=0,vmax=100,cmap='Blues')
    
    xt = np.linspace(0,sum(xdx)-1,sum(xdx))
    
    plt.yticks(xt,labels=u_t[xdx],fontsize=7)
    plt.xticks(xt,labels=u_t[xdx],rotation=90,fontsize=7)
    plt.title(dn)
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.2)
    plt.colorbar(label = 'Total synapses')
    
    plt.show()
    plt.rcParams['pdf.fonttype'] = 42
    savename = os.path.join(pltdir,'Multipartite'+ dn +'.png')
    plt.savefig(savename)
    savename = os.path.join(pltdir,'Multipartite'+ dn +'.pdf')
    plt.savefig(savename)
    
    
#%% 
nc_many = NC(bodyId=666308023,status='Traced', cropped=False)
syn = fetch_synapses(666308023,synapse_criteria=SC(rois='FB', primary_only=True,type='pre'))


many_conns = fetch_synapse_connections(666308023,synapse_criteria=SC(rois='FB', primary_only=True,type='post'))
many_conns = fetch_synapse_connections(666308023,synapse_criteria=SC(primary_only=True,type='post'))
many_conns = fetch_synapse_connections(target_criteria=666308023)


#%% Similar to above, but look for every post synapse if that also synapses onto another neuron
da_neurons = ['FB4M','FB4L','FB5H','FB6H','FB7B','FB2A']


for dn in range(len(da_neurons)):
    neuron_criteria = NC(status='Traced', type=da_neurons[dn], cropped=False)
    da_criteria = SC(rois='FB', type='pre', primary_only=True)
    da_syns = fetch_synapses(neuron_criteria, da_criteria)
    
    da_conns = fetch_synapse_connections(neuron_criteria, None, da_criteria)
    
    n_id = da_syns['bodyId'].to_numpy()
    nu = np.unique(n_id)
    #plt.scatter(da_syns['x'][n_id==nu[1]],da_syns['y'][n_id==nu[1]],s = 10)
    
    
    # Get index of post_types
    post_neurons, _ = fetch_neurons(da_conns['bodyId_post'].unique())
    u_t = post_neurons['type'].unique()
    u_n = post_neurons['bodyId']
    
    
    
    
    nc_many = NC(bodyId=da_conns['bodyId_post'].unique())
    
    
    pre_synloc =  da_syns[['x','y','z']].to_numpy()
    pre_syn_long = da_conns[['x_pre','y_pre','z_pre']]
    post_syn_long = da_conns[['x_post','y_post','z_post']]
    da_conId = da_conns['bodyId_post']
    roi_thresh = 0.5/0.008 # assuming 8nm resolution
    pair_mat = np.zeros((len(u_t),len(u_t))) # DA-Pre-post (pre synaptic plasticity)
    pair_mat_post = np.zeros_like(pair_mat) # DA-post-pre (post synaptic plasticity)
    type_list = u_t.copy()
    type_list_post = u_t.copy()
    sc = SC(rois='FB', primary_only=True)
    for i,u in enumerate(u_n):
        u_type = post_neurons['type'][i]
        print(i,' of ', len(u_n))
        if u_type is None:
           # print('None')
            continue
        
        try : 
            nc_many = NC(bodyId=u,status='Traced', cropped=False)
            many_conns = fetch_synapse_connections(u,synapse_criteria=SC(rois='FB', primary_only=True,type='pre'))
            many_conns_post = fetch_synapse_connections(target_criteria=u,synapse_criteria=SC(rois='FB', primary_only=True,type='post'))
        except:
            print('NP fail')
            continue
         
        many_conn_pre = many_conns[['x_pre','y_pre','z_pre']].to_numpy()
        
        many_conn_post = many_conns_post[['x_post','y_post','z_post']].to_numpy()
        
        
        # Iterate through synapses
        syn_dx = da_conId==u
        t_locs = post_syn_long[syn_dx].to_numpy()
        if many_conn_pre.size!=0:
            for t in t_locs:
                t_diff = many_conn_pre-t
                t_diff = np.sqrt(np.sum(t_diff**2,axis=1))
                if min(t_diff)<roi_thresh:
                    td = np.where(t_diff<roi_thresh)[0]
                    t_rois = many_conns['bodyId_post'][td].to_numpy()
                    p_neurons, _ = fetch_neurons(t_rois)
                    types = p_neurons['type'].to_numpy()
                    types[types==None] = 'No'
                    
                    for i2,t2 in enumerate(types):
                        t_type = types[i2]
                        t_id =p_neurons['bodyId'][i2]
                        if t_type!='No':
                            print(t_type)
                            if np.sum(type_list==t_type)>0:
                                type_dx = type_list==t_type
                                pair_mat[u_t==u_type,type_dx] = pair_mat[u_t==u_type,type_dx] +np.sum(many_conns['bodyId_post']==t_id)
                            else :
                                print('New Neuron')
                                type_list = np.append(type_list,t_type)
                                pair_mat = np.append(pair_mat,np.zeros((len(u_t),1)),axis=1)
                                pair_mat[u_t==u_type,-1] = pair_mat[u_t==u_type,-1]+np.sum(many_conns['bodyId_post']==t_id)
                            
                t_diff = many_conn_post-t
                t_diff = np.sqrt(np.sum(t_diff**2,axis=1))
                if many_conn_post.size!=0:
                    if min(t_diff)<roi_thresh:
                        td = np.where(t_diff<roi_thresh)[0]
                        t_rois = many_conns_post['bodyId_pre'][td].to_numpy()
                        p_neurons, _ = fetch_neurons(t_rois)
                        types = p_neurons['type'].to_numpy()
                        types[types==None] = 'No'
                        
                        for i2,t2 in enumerate(types):
                            t_type = types[i2]
                            
                            t_id =p_neurons['bodyId'][i2]
                            if t_type!='No' and t_type!='FB4M':
                                print(t_type)
                                if np.sum(type_list_post==t_type)>0:
                                    type_dx = type_list_post==t_type
                                    pair_mat_post[u_t==u_type,type_dx] = pair_mat_post[u_t==u_type,type_dx] +np.sum(many_conns_post['bodyId_pre']==t_id)
                                else :
                                    print('New Neuron')
                                    type_list_post = np.append(type_list_post,t_type)
                                    pair_mat_post = np.append(pair_mat_post,np.zeros((len(u_t),1)),axis=1)
                                    pair_mat_post[u_t==u_type,-1] = pair_mat_post[u_t==u_type,-1]+np.sum(many_conns_post['bodyId_pre']==t_id)
                    
                
                
    # Get neuron abundancies
    
    p_neurons, _ = fetch_neurons(NC(status='Traced',type=type_list[type_list!=None]))
    type_list[type_list==None] = 'None'
    pair_mat_norm = pair_mat.copy()
    for i,t in enumerate(u_t):
        if t!='None':
            pair_mat_norm[i,:] =  pair_mat_norm[i,:]/np.sum(p_neurons['type']==t)   
    pair_mat_norm[np.isnan(pair_mat_norm)] = 0
    
    pair_mat_post_norm = pair_mat_post.copy()
    for i,t in enumerate(u_t):
        if t!='None':
            pair_mat_post_norm[i,:] =  pair_mat_post_norm[i,:]/np.sum(p_neurons['type']==t)   
    pair_mat_post_norm[np.isnan(pair_mat_post_norm)] = 0
    
    savedict = {'pair_mat_pre':pair_mat_norm,'pair_mat_post':pair_mat_post_norm,
                'DA_neuron':da_neurons[dn],'post_DA_neuron_type':u_t,'pre_neuron_type':type_list_post,
                'post_neuron_type': type_list}
    savedir = os.path.join("Y:\\Data\\Connectome\\Connectome Mining\\DopamineSynAnalysis",da_neurons[dn]+ ".pickle")
    with open(savedir, "wb") as pickle_file:
        pkl.dump(savedict, pickle_file)
#%% 
plt.close('all')
pltdir = 'Y:\Presentations\\2024\\MarchLabMeeting\\Figures'
for i,d in enumerate(da_neurons) :
    savedir =os.path.join("Y:\\Data\\Connectome\\Connectome Mining\\DopamineSynAnalysis",d+ ".pickle")
    with open(savedir, "rb") as input_file:
        data = pkl.load(input_file)
    pair_mat = data['pair_mat_pre']
    pair_mat_norm = data['pair_mat_pre']
    u_t = data['post_DA_neuron_type']
    type_list = data['post_neuron_type']
    
    
    plt.figure()
    plt.rcParams['pdf.fonttype'] = 42 
    xdx = np.max(pair_mat,axis=1)>50
    ydx =np.max(pair_mat,axis=0)>50
    xt = np.linspace(0,sum(xdx)-1,sum(xdx))
    plt_mat = pair_mat_norm.copy()
    plt_mat = plt_mat[xdx,:]
    plt_mat = plt_mat[:,ydx]
    plt.imshow(plt_mat,aspect='auto',interpolation='none',vmin=0,vmax=100,cmap='Blues')
    plt.yticks(xt,labels=u_t[xdx])
    yt = np.linspace(0,sum(ydx)-1,sum(ydx))
    plt.xticks(yt,labels=type_list[ydx],rotation=90)
    plt.ylabel('Presynaptic neuron')
    plt.xlabel('Post-synaptic neuron')
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.2)
    plt.colorbar(label = 'Synapses per neuron')
    plt.title(d)
    plt.show()
    plt.rcParams['pdf.fonttype'] = 42
    savename = os.path.join(pltdir,'DA_Pre_post_'+ d +'.png')
    plt.savefig(savename)
    savename = os.path.join(pltdir,'DA_Pre_post_'+ d +'.pdf')
    plt.savefig(savename)
    
    
    
    pair_mat = data['pair_mat_post']
    pair_mat_norm = data['pair_mat_post']
    u_t = data['post_DA_neuron_type']
    type_list = data['pre_neuron_type']
    plt.figure()
    plt.rcParams['pdf.fonttype'] = 42 
    xdx = np.max(pair_mat,axis=1)>50
    ydx =np.max(pair_mat,axis=0)>50
    xt = np.linspace(0,sum(xdx)-1,sum(xdx))
    plt_mat = pair_mat_norm.copy()
    plt_mat = plt_mat[xdx,:]
    plt_mat = plt_mat[:,ydx]
    plt.imshow(plt_mat,aspect='auto',interpolation='none',vmin=0,vmax=100,cmap='Blues')
    plt.yticks(xt,labels=u_t[xdx])
    yt = np.linspace(0,sum(ydx)-1,sum(ydx))
    plt.xticks(yt,labels=type_list[ydx],rotation=90)
    plt.ylabel('Post-synaptic neuron')
    plt.xlabel('Pre-synaptic neuron')
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.2)
    plt.colorbar(label = 'Synapses per neuron')
    plt.title(d)
    plt.show()
    plt.rcParams['pdf.fonttype'] = 42
    savename = os.path.join(pltdir,'DA_Post_pre'+ d +'.png')
    plt.savefig(savename)
    savename = os.path.join(pltdir,'DA_Post_pre'+ d +'.pdf')
    plt.savefig(savename)



# plt.figure()
# plt.imshow(pair_mat_post,aspect='auto',interpolation='none',vmin=0,vmax=20)
# xdx = np.max(pair_mat_post,axis=1)>20
# xt = np.linspace(0,len(u_t)-1,len(u_t))
# plt.yticks(xt,labels=u_t)
# plt.xticks(xt,labels=u_t,rotation=90)
# plt.ylabel('Post-synaptic neuron')
# plt.xlabel('Pre-synaptic neuron')