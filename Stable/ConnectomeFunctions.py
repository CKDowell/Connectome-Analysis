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

def input_output_matrix(nname):
    criteria = NC(type = nname)
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
                
def hier_cosine(indata,distance_thresh):
    in_shape = np.shape(indata)
    
    sim_mat = np.empty([in_shape[0], in_shape[0]])
    ilen = int(in_shape[0])
    for i in range(in_shape[0]):
        x = indata[i,:]
        for z in range(in_shape[0]):
            y = indata[z,:]
            sim_mat[i,z] = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

    d_mat = 1-sim_mat

    cluster = AgglomerativeClustering(affinity='precomputed', linkage='single',
          
                                    compute_distances = True, distance_threshold =distance_thresh, n_clusters = None)
    cluster.fit(d_mat)
    return cluster, d_mat