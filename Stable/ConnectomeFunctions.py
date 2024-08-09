# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:30:56 2023

@author: dowel
"""

from neuprint import Client
import numpy as np
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
from neuprint import fetch_neurons, fetch_adjacencies, NeuronCriteria as NC
import pandas as pd
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import leaves_list
from Stable.SimulationFunctions import sim_functions as sf
#%%
def top_inputs(names):
    # Gets names of top input types
    criteria = NC(type=names)
    neuron_df, conn_df = fetch_adjacencies(None, criteria)
    prenames = conn_df['bodyId_pre']
    weights = conn_df['weight']
    idlib = neuron_df['bodyId']
    typelib = neuron_df['type']
    idlib_n = pd.Series.to_numpy(idlib)
    typelib = pd.Series.to_numpy(typelib,'str')
    typelib_u = np.unique(typelib)
    t_inputs = np.empty(len(typelib_u))
    ncells = np.empty(len(typelib_u))
    weights_n = pd.Series.to_numpy(weights)
    prenames_n = pd.Series.to_numpy(prenames)
    
    for i, t in enumerate(typelib_u):
        ma_n = typelib==t
        #print(t)
        tids = idlib_n[ma_n]
        id_idx = np.in1d(prenames_n,tids)
        #print(np.sum(weights_n(ma)))
        t_inputs[i] = np.sum(weights_n[id_idx])
        ncells[i] = np.sum(ma_n)
    return typelib_u, t_inputs, ncells

def top_outputs(names):
    criteria = NC(type=names)
    neuron_df, conn_df = fetch_adjacencies(criteria,None)
    postnames = conn_df['bodyId_post']
    weights = conn_df['weight']
    idlib = neuron_df['bodyId']
    typelib = neuron_df['type']
    idlib_n = pd.Series.to_numpy(idlib)
    typelib = pd.Series.to_numpy(typelib,'str')
    typelib_u = np.unique(typelib)
    t_outputs = np.empty(len(typelib_u))
    ncells = np.empty(len(typelib_u))
    weights_n = pd.Series.to_numpy(weights)
    postnames_n = pd.Series.to_numpy(postnames)
    
    for i, t in enumerate(typelib_u):
        ma_n = typelib==t
        #print(t)
        tids = idlib_n[ma_n]
        id_idx = np.in1d(postnames_n,tids)
        #print(np.sum(weights_n(ma)))
        t_outputs[i] = np.sum(weights_n[id_idx])
        ncells[i] = np.sum(ma_n)
    return typelib_u, t_outputs, ncells

def defined_in_out(names1,names2):
    # Access neuprint data
    
    neuron_df, conn_df = fetch_adjacencies(NC(type=names1),NC(type=names2))
    nd_simple,c_simple = fetch_neurons(NC(type=names1))
    out_simple,co_simple = fetch_neurons(NC(type=names2))
    # Get Pre-syn types
    types_u = np.unique(nd_simple['type'])
    #Get post-syn types
    typesO_u = np.unique(out_simple['type'])
    # Get NT identity
    S = sf()
    nt_dict = S.set_neur_dicts(types_u)
    
    con_mat_full = np.zeros([len(types_u), len(out_simple['type'])],'float64')
    con_mat_full_sign = np.zeros([len(types_u), len(out_simple['type'])],'float64')
    
    for i,t in enumerate(types_u):
        bids = nd_simple['bodyId'][nd_simple['type']==t]
        dx = np.in1d(conn_df['bodyId_pre'],bids)
        
        bid_post = conn_df['bodyId_post'][dx]
        
        weights = pd.Series.to_numpy(conn_df['weight'][dx])
        
        for ib,b in enumerate(bid_post):
            bdx = out_simple['bodyId']==b
            con_mat_full[i,bdx] = con_mat_full[i,bdx]+weights[ib]
        
        ns = nt_dict['NT_sign'][i]
        con_mat_full_sign[i,:] = con_mat_full[i,:]*ns
        
    con_mat = np.zeros([len(types_u), len(typesO_u)],'float64')
    con_mat_sign = np.zeros([len(types_u), len(typesO_u)],'float64')
    con_mat_sum = np.zeros([len(types_u), len(typesO_u)],'float64')
    type_count = np.array([])
    for i, t in enumerate(typesO_u):
        dx = out_simple['type']==t
        type_count = np.append(type_count,np.sum(dx))
        con_mat[:,i] = np.mean(con_mat_full[:,dx],axis=1).flatten()
        con_mat_sign[:,i] = np.mean(con_mat_full_sign[:,dx],axis=1).flatten()
        con_mat_sum[:,i] = np.sum(con_mat_full[:,dx],axis=1).flatten()
    out_dict = {'in_types':types_u,'out_types': typesO_u,'con_mat': con_mat,
                'con_mat_sum': con_mat_sum,
                'con_mat_sign': con_mat_sign, 'con_mat_full': con_mat_full,
                'con_mat_full_sign': con_mat_full_sign}
    
    return out_dict
    
def defined_in_out_full(names1,names2):
    neuron_df, conn_df = fetch_adjacencies(NC(type=names1),NC(type=names2))
    #nd_simple,c_simple = fetch_neurons(NC(type=names1))
    #out_simple,co_simple = fetch_neurons(NC(type=names2))
    
    conmat = np.zeros((np.shape(neuron_df)[0],np.shape(neuron_df)[0]))
    nids = neuron_df['bodyId']
    pre_ids = conn_df['bodyId_pre'].to_numpy()
    post_ids = conn_df['bodyId_post'].to_numpy()
    cons = conn_df['weight'].to_numpy()
    for i, n in enumerate(pre_ids):
        dx1 = np.where(nids==n)[0]
        dx2 = np.where(nids==post_ids[i])[0]
        conmat[dx1,dx2] = cons[i]
    
    types_u = np.unique(neuron_df['type'])
    
    
    
    # Get NT identity
    S = sf()
    nt_dict = S.set_neur_dicts(types_u)
    conmat_sign = conmat.copy()
    for i,n in enumerate(types_u):
        nt_sign = nt_dict['NT_sign'][i]
        dx = neuron_df['type']==n
        conmat_sign[dx,:] = conmat_sign[dx,:]*nt_sign
    out_dict = {'conmat':conmat,'conmat_sign':conmat_sign,'NTdict': nt_dict,'nDF': neuron_df}
    return out_dict 
    
def input_output_matrix(names):
    criteria = NC(type = names)
    neuron_df, roi_counts_df = fetch_neurons(criteria)
    types = neuron_df['type']
    types = pd.Series.to_numpy(types)
    types_u = np.unique(types)

    
    for i, t in  enumerate(types_u):
        print(str(i+1) + '/' + str(len(types_u)))
        
        # Inputs
        typelib, t_inputs, ncells = top_inputs(t)
        if i==0:
            in_types = typelib
            in_array = np.empty([len(types_u), len(t_inputs)])
            in_array[i,:] = np.transpose(t_inputs)
            
        else:
            for r, t2 in enumerate(typelib):
                tdx = in_types==t2
                if sum(tdx)>0:
                    in_array[i,tdx] = t_inputs[r]
                else:
                    in_types = np.append(in_types,typelib[r])
                    add_array = np.zeros([len(types_u),1])
                    in_array = np.append(in_array,add_array,axis=1)
                    in_array[i,-1:] = t_inputs[r]
                    
            if sum(in_array[i,:])<0:
                print('In array negative')
                
        # Outputs
        typelib, t_outputs, ncells = top_outputs(t)
        if i==0:
            out_types = typelib
            out_array = np.empty([len(types_u), len(t_outputs)])
            out_array[i,:] = np.transpose(t_outputs)
            
        else:
            for r, t2 in enumerate(typelib):
                tdx = out_types==t2
                if sum(tdx)>0:
                    out_array[i,tdx] = t_outputs[r]
                else:
                    out_types = np.append(out_types,typelib[r])
                    add_array = np.zeros([len(types_u),1])
                    out_array = np.append(out_array,add_array,axis=1)
                    out_array[i,-1:] = t_outputs[r]
    return out_types, in_types, in_array, out_array, types_u

def con_matrix_iputs(names):
    out_types, in_types, in_array, out_array, types_u = input_output_matrix(names)
    neuron_df, conn_df = fetch_adjacencies(NC(type=in_types), NC(type=in_types))
    con_matrix = np.zeros([len(in_types), len(in_types)])

    # Prefer to use a for loop so that I absolutely know which is which in the right order

    for i,t in enumerate(in_types):
        print(i)
        t1dx = neuron_df['type']==t
        t1s = pd.Series.to_numpy(neuron_df['bodyId'][t1dx])
        t1c_dx = np.in1d(pd.Series.to_numpy(conn_df['bodyId_pre']),t1s)
        
        for i2,t2 in enumerate(in_types):
            t2dx = neuron_df['type']==t2
            t2s = pd.Series.to_numpy(neuron_df['bodyId'][t2dx])
            t2c_dx = np.in1d(pd.Series.to_numpy(conn_df['bodyId_post']),t2s)
            dx = t1c_dx&t2c_dx
            ws = pd.Series.to_numpy(conn_df['weight'][dx])
            con_matrix[i,i2] = np.sum(ws)
    return con_matrix
                
def hier_cosine(indata,distance_thresh):
    in_shape = np.shape(indata)
    
    sim_mat = np.zeros([in_shape[0], in_shape[0]],dtype='float64')
    ilen = int(in_shape[0])
    for i in range(in_shape[0]):
        x = indata[i,:]
        for z in range(in_shape[0]):
            y = indata[z,:]
            sim_mat[i,z] = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
            if np.isnan(sim_mat[i,z]):
                print('i',i)
                print('z',z)
    d_mat = 1-sim_mat

    cluster = AgglomerativeClustering(linkage='single', 
                                    compute_distances = True, distance_threshold =distance_thresh, n_clusters = None)
    cluster.fit(d_mat)
    return cluster, d_mat

def linkage_order(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    z = leaves_list(linkage_matrix)
    return z