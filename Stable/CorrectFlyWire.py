# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:57:52 2024

@author: dowel
"""

#%% Motivation
# Flywire is annotated with neuron names from Neuprint but there is not a 100%
# correspondence.
# The allocation is based only on anatomy via the NBLAST method, which has its
# limitations. This script aims to match poorly assigned neurons to Neuprint
# via the similarity of their inputs and outputs. 

#%% Algorithm outline
# 1. Choose unallocated neuron type
# 2. Get inputs and outputs from Neuprint
# 3. Compare to FlyWire using two metrics
# - Number of shared pre and post partners
# - Spearman/Pearson's correlation of pre and post partners

# 4. Output is an updated name with the two metrics highlighted

# 5. Larger aim is to do this for every unallocated neuron and output a CSV file
# with the corrected names
# 6. Things to watch out for: there are definitely Flywire neurons that do not
# exist in neuprint. There should not be some misallocation here. Also it may
# be important to take cell body location into account
#%%
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
import scipy as sp
from Stable.ConnectomeFunctions import input_output_matrix
from scipy import stats
import matplotlib.pyplot as plt
from neuprint import Client
import numpy as np
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
from neuprint import fetch_neurons, NeuronCriteria as NC
#%%
class fw_corrections:
    def __init__(self,classtype='original'):
        self.datapath = "D:\ConnectomeData\FlywireWholeBrain"
        print('Loading Flywire data')
        
        # Connections
        dpath = os.path.join(self.datapath ,'connections_princeton_no_threshold.csv')
        self.connections = pd.read_csv(dpath)#read in
        
        # Neuron classification
        if classtype=='original':
            dpath = os.path.join(self.datapath ,'classification.csv')
        elif classtype=='Charlie':
            dpath = os.path.join(self.datapath ,'classification_Charlie.csv')
        self.classes = pd.read_csv(dpath)
        dpath = os.path.join(self.datapath,'consolidated_cell_types.csv')
        self.flywire_types = pd.read_csv(dpath)
        # Neuron summaries
        dpath = os.path.join(self.datapath,'neurons.csv')
        self.neurons = pd.read_csv(dpath)
        print('Flywire data loaded')    
        # Loaded flywire data
        
    def allocate_by_connections(self,NP_neuron,FW_neuron,namingconvention='hemibrain_type'):
        # Standard variables
        
        # Neuprint connections
        out_types, in_types, in_array, out_array, types_u,tcounts = input_output_matrix(NP_neuron)
        
        # Query neuprint for
        c = self.classes.copy() 
        o_pred = pd.Series.to_numpy(c[namingconvention])
        
        # Check if already allocated
        check = np.sum(o_pred==FW_neuron)
        if check>0:
            print('Neuron already allocated in flywire!!')
            return 
        
        # Check if partial allocation
        dx_neuron = np.array([],dtype=int)
        nids = pd.Series.to_numpy(c['root_id'])
        
        for i,p in enumerate(o_pred):
            if pd.isnull(p):
                continue
            if FW_neuron in p:

                dx_neuron = np.append(dx_neuron,i)
        if len(dx_neuron)>0:
            print('Multi-type prediction in flywire!')
            print('Narrowing it down...')
            # Check connectivity
            # Get flywire connectivity matrix
            
            print(nids[dx_neuron])
            c_mat,df_meta = self.get_conmat(nids[dx_neuron])
            # Get connections to known types
            ht = pd.Series.to_numpy(df_meta['hemibrain_type'])
            dxkeep = np.array([],dtype=int)
            for i,t in enumerate(ht):     
                if pd.isnull(t):
                    continue
                elif not (',' in t):
                    dxkeep = np.append(dxkeep,i)
                    print(t)
            
            # Overlap in pre and post populations with flywire
            in_out_types  = np.unique(ht[dxkeep])
            in_overlap = in_types[np.in1d(in_types,in_out_types) ]
            out_overlap = out_types[np.in1d(out_types,in_out_types)]
            
            dx_in_np  = np.in1d(in_types,in_overlap)
            dx_out_np = np.in1d(out_types,out_overlap)
            np_invec = in_array[:,dx_in_np]
            np_outvec = out_array[:,dx_out_np]
            
            #
            
           
            # Our neurons of interest in flywire        
            nids = pd.Series.to_numpy(self.classes['root_id'].copy())
            our_ndx = np.in1d(df_meta['root_id'],nids[dx_neuron])
            our_ndx = np.where(our_ndx)[0]
            # output matrix flywire across types
            out_mat_full = c_mat[our_ndx,:]
            out_mat = np.zeros([len(our_ndx),len(out_overlap)])
            for i,o in enumerate(out_overlap):
                odx = ht==o
                out_mat[:,i] = np.sum(out_mat_full[:,odx],axis=1).flatten()

            # input matrix flywire across types
            in_mat_full = c_mat[:,our_ndx]
            in_mat = np.zeros([len(in_overlap),len(our_ndx)])
            for i,o in enumerate(in_overlap):
                odx = ht==o
                in_mat[i,:] = np.sum(in_mat_full[odx,:],axis=0).flatten()
            
            
            in_mat = np.transpose(in_mat)
            # Sort outputs by Neuprint
            corr_pearson = np.zeros([len(our_ndx),2])
            corr_spearman = np.zeros([len(our_ndx),2])
            
            # Calculate correlations to input and output vectors
            for i in range(len(our_ndx)):   
                # output
                sp = stats.spearmanr(np_outvec[0,:],out_mat[i,:])
                pr = stats.pearsonr(np_outvec[0,:],out_mat[i,:])
                corr_spearman[i,0] = sp.statistic
                corr_pearson[i,0] = pr.statistic
                # input
                sp = stats.spearmanr(np_invec[0,:],np.transpose(in_mat[i,:]))
                pr = stats.pearsonr(np_invec[0,:],np.transpose(in_mat[i,:]))
                corr_spearman[i,1] = sp.statistic
                corr_pearson[i,1] = pr.statistic
                o_nids = pd.Series.to_numpy(df_meta['root_id'])
                our_ids = o_nids[our_ndx]
        else:
            print('No multi-type prediction...')
            print('Searching full connectome')
            df_meta = self.get_conmat(nids,return_meta=True)
            
            ht = pd.Series.to_numpy(df_meta['hemibrain_type'])
            dxkeep = np.array([],dtype=int)
            for i,t in enumerate(ht):     
                if pd.isnull(t):
                    continue
                elif not (',' in t):
                    dxkeep = np.append(dxkeep,i)
                    #print(t)
            
            ht_small = ht[dxkeep]
            nidsmall = nids[dxkeep]
            # Overlap in pre and post populations with flywire
            in_out_types  = np.unique(ht[dxkeep])
            in_overlap = in_types[np.in1d(in_types,in_out_types) ]
            out_overlap = out_types[np.in1d(out_types,in_out_types)]
            
            dx_in_np  = np.in1d(in_types,in_overlap)
            dx_out_np = np.in1d(out_types,out_overlap)
            np_invec = in_array[:,dx_in_np]
            np_outvec = out_array[:,dx_out_np]
            
            ht_overlap_dx = np.logical_or(np.in1d(ht_small, in_overlap),np.in1d(ht_small, out_overlap))
            
            tnids = nidsmall[ht_overlap_dx]
            
            nidx = np.logical_and(np.logical_or(np.in1d(self.connections['pre_pt_root_id'],tnids)
                                 ,np.in1d(self.connections['post_pt_root_id'],tnids)),
                                  self.connections['syn_count']>4)
            
            nidsall,counts = np.unique(np.append(self.connections['pre_pt_root_id'][nidx].to_numpy(),
                                          self.connections['post_pt_root_id'][nidx].to_numpy()
                                          ),return_counts=True)
            print('Searching ',len(nidsall),' neurons')
            #nidsmall =nidsp
       
            out_mat_full,df_meta = self.get_conmat(nidsall,exclusive=True)
            #our_ndx = np.in1d(df_meta['root_id'],nids[dx_neuron])
            
            # Running this again since we are cutting down the types to those with >4 synapses
            ht = pd.Series.to_numpy(df_meta['hemibrain_type'])
            dxkeep = np.array([],dtype=int)
            for i,t in enumerate(ht):     
                if pd.isnull(t):
                    continue
                elif not (',' in t):
                    dxkeep = np.append(dxkeep,i)
            ht= ht[dxkeep]
            
            in_out_types  = np.unique(ht)
            in_overlap = in_types[np.in1d(in_types,in_out_types) ]
            out_overlap = out_types[np.in1d(out_types,in_out_types)]
            
            dx_in_np  = np.in1d(in_types,in_overlap)
            dx_out_np = np.in1d(out_types,out_overlap)
            np_invec = in_array[:,dx_in_np]
            np_outvec = out_array[:,dx_out_np]
            
            
            
            our_ndx = np.arange(0,len(nidsall))
            # output matrix flywire across types
            #out_mat_full = c_mat[our_ndx,:]
            out_mat = np.zeros([len(our_ndx),len(out_overlap)])
            for i,o in enumerate(out_overlap):
                odx = ht==o
                out_mat[:,i] = np.sum(out_mat_full[:,odx],axis=1).flatten()

            # input matrix flywire across types
            in_mat_full = out_mat_full.copy()
            in_mat = np.zeros([len(in_overlap),len(our_ndx)])
            for i,o in enumerate(in_overlap):
                odx = ht==o
                in_mat[i,:] = np.sum(in_mat_full[odx,:],axis=0).flatten()
            
            
            in_mat = np.transpose(in_mat)
            # Sort outputs by Neuprint
            corr_pearson = np.zeros([len(our_ndx),2])
            corr_spearman = np.zeros([len(our_ndx),2])
            
            # Calculate correlations to input and output vectors
            for i in range(len(our_ndx)):   
                # output
                sp = stats.spearmanr(np_outvec[0,:],out_mat[i,:])
                pr = stats.pearsonr(np_outvec[0,:],out_mat[i,:])
                corr_spearman[i,0] = sp.statistic
                corr_pearson[i,0] = pr.statistic
                # input
                sp = stats.spearmanr(np_invec[0,:],np.transpose(in_mat[i,:]))
                pr = stats.pearsonr(np_invec[0,:],np.transpose(in_mat[i,:]))
                corr_spearman[i,1] = sp.statistic
                corr_pearson[i,1] = pr.statistic
                o_nids = pd.Series.to_numpy(df_meta['root_id'])
                our_ids = o_nids[our_ndx]
            
            
            
            
            
            print('Broader search under development... stay tuned...')
            #return
        
        
        # Now rank by top inputs and outputs and choose number seen in Neuprint
        corr_met = np.sum(corr_pearson,axis=1)
        criteria = NC(type = NP_neuron)
        neuron_df, roi_counts_df = fetch_neurons(criteria)
        num_neurons = np.shape(neuron_df)[0]
        crnk = np.argsort(-corr_met)
        candis = crnk[:num_neurons]
        cand_ids = our_ids[candis] 
        # Add additional neurons that come very close to these values
        output = {'Corr_pearson': corr_pearson,'Corr_spearman': corr_spearman,
                   'our_ids': our_ids,'corr_metric': corr_met,'top_candidates': cand_ids,'top_pearson': corr_pearson[candis,:],
                   'top_corr_metric': corr_met[candis]}
        return output
        
            
    def get_conmat(self,n_number,return_meta=False,exclusive=False):
        df = self.connections.copy()
        df_meta = self.classes.copy() 

        df = df.groupby(['pre_pt_root_id', 'post_pt_root_id']).agg({'syn_count': 'sum', 'neuropil':'first', 
               'nt_type':'first'}).reset_index()#sum synapses across all unique pre and post pairs

        # Set up indicies
        pre = df['pre_pt_root_id']
        
        dx1 = np.isin(pre,n_number)
        
        post = df['post_pt_root_id']
        dx2 = np.in1d(post,n_number)
        if exclusive:
            dx = np.logical_and(dx1,dx2)
        else:
            dx = np.logical_or(dx1,dx2)
    
        dx_i = np.where(dx)[0]    
        #dx_i = [i for i,r in enumerate(dx.astype(int)) if r>0.1]
        df_small = df.iloc[dx_i]


        vals, inds, inv = np.unique(list(df_small['pre_pt_root_id'].values) + list(df_small['post_pt_root_id'].values), return_index=True, return_inverse=True)
        conv_dict = {val:i for i, val in enumerate(vals)}
        df_small['pre_root_id_unique'] = [conv_dict[val] for val in df_small['pre_pt_root_id']]
        df_small['post_root_id_unique'] = [conv_dict[val] for val in df_small['post_pt_root_id']]
        n = len(vals)#total rows of full dynamics matrix
        syn_count = df_small['syn_count'].values
        pre_root_id_unique = df_small['pre_root_id_unique'].values
        post_root_id_unique = df_small['post_root_id_unique'].values
        metdx = np.empty(np.shape(vals),dtype=int)
        #print(np.shape(vals))
        #vals_u = np.unique(np.append(df_small['pre_pt_root_id'].values,df_small['post_pt_root_id'].values))
        for i,v in enumerate(vals):
            #print('vee',v)
            #print(np.sum(df_meta['root_id']==v))
            metdx[i] = np.where(df_meta['root_id']==v)[0]
        #metdx = np.isin(df_meta['root_id'],vals)
        df_meta = df_meta.iloc[metdx]
        if return_meta:
            return df_meta
        
        # form unscaled sparse matrix
        C_orig = csr_matrix((syn_count, (pre_root_id_unique,post_root_id_unique)),shape=(n,n), dtype='float64')
        C_full = C_orig.todense()
        C_full = C_full
        return C_full, df_meta
    
    def update_class(self,n_number,assignment):
        # Update the assignment of neuron classifications
        # Be careful with this. If you iterate through updates errors will propagate
        print('Updating classifications, be careful!')
        c_old = self.classes
        dpath = os.path.join(self.datapath ,'classification_Charlie.csv')
        dpath2 = os.path.join(self.datapath,'ReassignmentTally.csv')
        c = pd.read_csv(dpath)
        cRA = pd.read_csv(dpath2)
        nids = pd.Series.to_numpy(self.classes['root_id'].copy())
        dx = np.in1d(nids,n_number)
        asvec = [assignment for n in n_number]
        print(asvec)
        d_add = {'root_id': pd.Series.to_numpy(c_old['root_id'][dx]), 'old_ass': pd.Series.to_numpy(c_old['hemibrain_type'][dx]),'new_ass': asvec}
        cRA_add = pd.DataFrame(d_add)
        cRA = cRA._append(cRA_add,ignore_index=True)
        cRA.to_csv(dpath2,index=False)
        c['hemibrain_type'][dx] = assignment
        c.to_csv(dpath)
        print('Updated')
#%% 
# thiscell = [720575940626316606,720575940632862177]
# fw = fw_corrections()
# c_full, df_meta = fw.get_conmat(thiscell)
# for c in thiscell:
#     dx_in = df_meta['root_id']==c
#     in_put = c_full[dx_in,:]
#     in_put = np.transpose(in_put)
#     plt.plot(np.linspace(0,len(in_put)-1,len(in_put)),in_put)
#     plt.xticks(np.linspace(0,len(in_put)-1,len(in_put)),labels=df_meta['hemibrain_type'],rotation=90)
#%%
# NP_neuron = 'FB4R'
# fw = fw_corrections()
# predictions = fw.allocate_by_connections(NP_neuron)   
# ass_thresh = 1.6
# pred_ids = predictions['top_candidates']
# pred_met = predictions['top_corr_metric']
# keep = pred_met>ass_thresh
# assign_ids = pred_ids[keep]
# fw.update_class(assign_ids,NP_neuron)
   
