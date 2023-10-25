# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:21:55 2023

@author: dowel
"""

#%% 
from neuprint import Client
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
import pandas as pd
import numpy as np
from neuprint import fetch_adjacencies, fetch_neurons, NeuronCriteria as NC
from scipy import stats
import pickle
import os
import pyarrow as pa
import pyarrow.feather as feather
import bz2

#%% 
class sim_functions:
    def __init__(self):
        self.NTfilepath = 'D:\\ConnectomeData\\Neurotransmitters\\' 
#%% 
    def get_NT_identity(self,body_ids):
        print('Loading NT dataset')
        filepath = os.path.join(self.NTfilepath,'hemibrain-v1.2-tbar-neurotransmitters.feather.bz2')
    
        with bz2.open(filepath, 'rb') as bz2_file:
            # Read the decompressed binary data
            decompressed_data = bz2_file.read()
    
        # Read the Feather data from the decompressed binary data
        df = feather.read_feather(pa.py_buffer(decompressed_data))
        #df = feather.read_feather(filepath)
        print('Loaded')
        df_names = df.head(0)
        nt_list = [i for i in df_names if 'nts_8' in i]
        NT_matrix = np.zeros([len(body_ids),len(nt_list)])
        syn_body_ids = pd.Series.to_numpy(df['body'])
        for i, bod in enumerate(body_ids):
            bodx = syn_body_ids==bod
            for n, nt in enumerate(nt_list):
                t_nt = df[nt][bodx]
               
                NT_matrix[i,n] = np.sum(t_nt)
        nt_id = np.argmax(NT_matrix,1)
        print('Got NTs')
        # NT list 0: gaba 1: ACh 2: Glu 3: 5HT 4: octopamine 5: DA 6: neither
        return NT_matrix, nt_id, body_ids
#%% Create a pkl file with NT identities - speeds up loading data
    def initialise_NTs(self):
        criteria = NC(type = '.*')
        neuron_df,roi_df = fetch_neurons(criteria)
        all_nt = self.get_NT_identity(neuron_df['bodyId'])

        savename = "All_NTs.pkl"
        savedir = self.NTfilepath

        savepath = os.path.join(savedir, savename)

        with open(savepath, 'wb') as file:
            pickle.dump(all_nt, file)
#%% Return neuron dictionary of properties
    def set_neur_dicts(self,types):
        print('Getting neurons')
        criteria = NC(type=types)
        df_neuron,roi_neuron = fetch_neurons(criteria)
        print('Getting NTs')
        #NT_matrix, nt_id = get_NT_identity(df_neuron['bodyId']) #Uncomment this if you want to compute NT identies from original synister data
        with open(os.path.join(self.NTfilepath, 'All_NTs.pkl'), 'rb') as f:
            All_NTs = pickle.load(f)
        nt_id = All_NTs[1]
        n_names = pd.Series.to_numpy(df_neuron['type'])
        n_ids = pd.Series.to_numpy(df_neuron['bodyId'])
        tan_names = np.unique(n_names)
        nt_sign = np.empty(np.shape(tan_names))
        nt_ids = np.empty(np.shape(tan_names))
        # Below sets sign of NT effect, edit if you think is wrong
        nt_sign_index = [-1, 1, -1, 1, 1, -1, 0]# -1 GABA, 1 ACh, -1 Glu, 1 5HT, 1 Oct, -1 DA, 0 NA
        for i,t in enumerate(tan_names):
            bod_dx = n_names==t
            t_ids = n_ids[bod_dx]
            n_dx = np.in1d(All_NTs[2],t_ids)
            n_types = nt_id[n_dx]
            nt_type = stats.mode(n_types)
            nt_sign[i] = nt_sign_index[nt_type.mode]
            nt_ids[i] = nt_type.mode
            
        NT_list = ['gaba', 'ACh', 'Glu', '5HT', 'Oct',  'DA', 'NA']
        neur_dicts = dict({'Neur_names':tan_names, 'NT_sign':nt_sign, 'NT_id':nt_ids, 'NT_list': NT_list})
        print('Done')
        return neur_dicts
#%% Runs simulation of neuron activation
    def run_sim_act(self,inputs,neurons,iterations):
        neur_dicts = self.set_neur_dicts(neurons) #gets types and associated NTs
        n_ids = np.empty(0,dtype='int64')
        n_types = np.empty(0)
        for n in neurons:
            criteria = NC(type=n)
            df_neuron,roi_neuron = fetch_neurons(criteria)    
            n_ids = np.append(n_ids,pd.Series.to_numpy(df_neuron['bodyId'],dtype='int64'))
            n_types = np.append(n_types,df_neuron['type'])
        criteria = NC(bodyId=n_ids)
        
        neurons_df, roi_con_df = fetch_adjacencies(sources = n_ids,targets = n_ids,batch_size=200)#get connections
        
        
        
        neuron_basic,roi_counts = fetch_neurons(n_ids)
        
        norm_dx = np.in1d(neuron_basic['bodyId'],neurons_df['bodyId'])
        
        norm_outweight = neuron_basic['post'][norm_dx] # There is also an upstream field that I think includes connections to unidentified neurites
        norm_outweight = np.expand_dims(norm_outweight,1)
        norm_outweight = np.transpose(norm_outweight)
        
        # Set up connectivity matrix - 
        conmatrix = np.zeros([len(neurons_df), len(neurons_df)])
        bod_id = pd.Series.to_numpy(neurons_df['bodyId'])
        con_in = pd.Series.to_numpy(roi_con_df['bodyId_pre'])
        con_out = pd.Series.to_numpy(roi_con_df['bodyId_post'])
        weights = pd.Series.to_numpy(roi_con_df['weight'])
        types = pd.Series.to_numpy(neurons_df['type'])
        for i,b in enumerate(bod_id):   
            indx = con_in==b
            t_outs = con_out[indx]
            t_w = weights[indx]
            
            t_outs_u = np.unique(t_outs) # annoyingly not all connections are summed as 1-2-1
            t_w_u = np.zeros_like(t_outs_u)
            for iu, u in enumerate(t_outs_u):
                todx = t_outs==u
                t_w_u[iu] = np.sum(t_w[todx])
            
            t_out_dx = np.where(np.in1d(bod_id,t_outs_u))
            conmatrix[i,t_out_dx] = t_w_u
        
        conmatrix = conmatrix/norm_outweight # normalise input strength by total input to each neuron
        # Set input weights
        w_types = neur_dicts['Neur_names']
        for r, i in enumerate(w_types):
            t_sign = neur_dicts['NT_sign'][r]
            tdx = types==i
            conmatrix[tdx,:] = conmatrix[tdx,:]*t_sign
            
        # Set up activation vector
        a_vec = np.zeros([len(types), 1])
        for i in inputs:
            idx = types==i
            a_vec[idx] = 1
        
        
        act_vec = np.transpose(a_vec)
        activity_mat = act_vec
        act_vec = np.matmul(act_vec,conmatrix)
        activity_mat = np.append(activity_mat,act_vec,axis=0)
        for i in range(iterations-1):
            act_vec[act_vec<0] = 0 # need to stop negative activity from being propagated, this effectively puts a relu in place
            act_vec = np.matmul(act_vec,conmatrix)
            activity_mat = np.append(activity_mat,act_vec,axis=0)
            
        
        u_types = np.unique(types)
        activity_mat_type = np.zeros([iterations+1,len(u_types),])
        for i, t in enumerate(u_types):
            tdx = types==t
            activity_mat_type[:,i] = np.mean(activity_mat[:,tdx],axis=1)
        
        sim_output = dict({'ActivityAll': activity_mat, 'TypesAll': types, 'ROI_ID': bod_id, 
                           'MeanActivityType': activity_mat_type, 'TypesSmall': u_types})    
        
        return sim_output