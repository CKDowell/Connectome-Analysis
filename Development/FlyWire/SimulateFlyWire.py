# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:28:09 2023

@author: dowel
"""

#%% Script summary
# To make a simple simulation of the flywire connectome
# I think may be best to construct in a more flexible way than was done for Neuprint
# It may be good to have a core simulation engine and sub functions that will assign input
# activity vectors to be propagated through the network

# Planned functions
# Initialise network - Choose network size, cells by type, region etc
# Activate cell class - activate a single class of neuron
# Activate specific neurons - activate with specific sets of neurons
# Simple plots of output
#%% Import packages

#%%
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
import scipy as sp
class FW_sim:
    def __init__(self):
        self.datapath = "D:\ConnectomeData\FlywireWholeBrain"
        print('Loading data')
        
        # Connections
        dpath = os.path.join(self.datapath ,'connections.csv')
        self.connections = pd.read_csv(dpath)#read in
        
        # Neuron classification
        dpath = os.path.join(self.datapath ,'classification.csv')
        self.classes = pd.read_csv(dpath)
        
        # Neuron summaries
        dpath = os.path.join(self.datapath,'neurons.csv')
        self.neurons = pd.read_csv(dpath)
        print('Data loaded, ready to sim!')
        
        # Neurotransmitter effects - may want to tweak
        self.neurotransmitter_effects = {
            'ACH': 1,   
            'DA': -1,     
            'GABA': -1,  
            'GLUT': -1,  
            'OCT': 1,   
            'SER': 1    
            }
        
        
    def initialise_network(self,choice,contype='small'):
        
        # Initialises the connectivity matrix with neurons of choice
        
        #Choose neurons
        print('Building connectivity matrix')
        if choice['Type']=='NeuronClass':
            
            if choice['Neuprint']: # Choose neurons by neuprint type
                all_ns = []
                for n in choice['NeuronClass']:
                    all_ns = np.append(all_ns,self.get_np_from_FW(n))
                
                all_ns = np.unique(all_ns)
                
            elif choice['FwClass']: # Choose by FlyWire class
                n_list = self.classes['root_id'].copy()
                ndx = self.classes['class'].copy()==choice['NeuronClass']
                n_number = n_list[ndx]
                
            elif choice['FwSuper']: # Choose by FlyWire super-class
                n_list = self.classes['root_id'].copy()
                ndx = self.classes['super_class'].copy()==choice['NeuronClass']
                n_number = n_list[ndx]
                
        elif choice['Type']=='NeuronNumber': # Choose by neuron index
            n_number = choice['NeuronNumber']
            
        elif choice['Type']=='Neuropil': # Choose by synapse neuropil location
            rois = self.connections['neuropil'].copy()
            pre = self.connections['pre_root_id'].copy()
            post = self.connections['post_root_id'].copy()
            pre = pd.Series.to_numpy(pre,dtype='int64')
            post = pd.Series.to_numpy(post,dtype='int64')
            
            
            all_rois = np.array([], dtype='int64')
            for n in choice['NeuronClass']:
                dx = rois==n
                n_1 = pre[dx]
                n_2 = post[dx]
                all_rois = np.append(all_rois,n_1)
                all_rois = np.append(all_rois,n_2)
                
            
            n_number = np.unique(all_rois)
            print(type(n_number[0]))
        elif choice['Type']=='All': # Choose all neurons
            n_number = self.classes['root_id']
        
        n_number = n_number
        self.n_number = n_number
        
        # Set up connectivity matrix - code taken from Pillow publication
        
        # Reorder matrices
        neurotransmitter_effects = self.neurotransmitter_effects
        df = self.connections.copy()
        df_meta = self.classes.copy()
        
        
        
        df['nt_sign'] = df['nt_type'].map(neurotransmitter_effects)#convert type to sign in the data frame
        df['syn_cnt_sgn'] = df['syn_count']*df['nt_sign']#multiply count by sign for unscaled 'effectome'
        df = df.groupby(['pre_root_id', 'post_root_id']).agg({'syn_cnt_sgn': 'sum', 'syn_count': 'sum', 'neuropil':'first', 
               'nt_type':'first'}).reset_index()#sum synapses across all unique pre and post pairs

        # Set up indicies
        n_number = n_number #have to do this because number is too large for index precision
        pre = df['pre_root_id']
        
        dx1 = np.isin(pre,n_number)
        
        post = df['post_root_id']
        dx2 = np.in1d(post,n_number)
        if contype=='small':# small restricts network to within selection
            dx = dx1&dx2
        elif contype =='large':# large includes all pre and post partners of selected neurons
            dx = dx1 or dx2
            
        dx_i = [i for i,r in enumerate(dx) if r>0.1]
        n_check = np.append(df['pre_root_id'][dx],df['post_root_id'][dx])
        n_check = np.unique(n_check)
        
        df_small = df.iloc[dx_i]


        vals, inds, inv = np.unique(list(df_small['pre_root_id'].values) + list(df_small['post_root_id'].values), return_index=True, return_inverse=True)
        conv_dict = {val:i for i, val in enumerate(vals)}
        df_small['pre_root_id_unique'] = [conv_dict[val] for val in df_small['pre_root_id']]
        df_small['post_root_id_unique'] = [conv_dict[val] for val in df_small['post_root_id']]
        n = len(vals)#total rows of full dynamics matrix
        syn_count_sgn = df_small['syn_cnt_sgn'].values
        pre_root_id_unique = df_small['pre_root_id_unique'].values
        post_root_id_unique = df_small['post_root_id_unique'].values
        
        # this order may not be right
        metdx = np.empty(np.shape(vals),dtype=int)
        for i,v in enumerate(vals):
            metdx[i] = np.where(df_meta==['root_id'],vals)
        #metdx = np.isin(df_meta['root_id'],vals)
        df_meta = df_meta.iloc[metdx]
        
        # form unscaled sparse matrix
        C_orig = csr_matrix((syn_count_sgn, (pre_root_id_unique,post_root_id_unique)), shape=(n, n), dtype='float64')
        # has pre neuron as row and post as 
        
        # need to normalise matrix
        col_divide = np.sum(np.abs(C_orig),axis=0)
        col_divide[col_divide==0] = 1 #gets rid of dividing by zero, will not make a difference 
        C_orig = C_orig/col_divide
        
        # dictionary to go back to original ids of cells
        conv_dict_rev = {v:k for k, v in conv_dict.items()}
        root_id = [conv_dict_rev[i] for i in range(n)]
        df_meta_W = df_meta.set_index('root_id', ).loc[root_id]
        df_meta_W['root_id_W'] = np.arange(n)
        df_meta_W = df_meta_W.set_index('root_id_W')
        df_meta_W['root_id'] = root_id
        
        # Output the connectivity matrix and meta data matrix aligned to con matrix
        self.C_mat = C_orig
        self.df_meta = df_meta_W
        print('Built')
        
    def get_np_from_FW(self,neuronclass):
        npvector = self.classes['hemibrain_type']
        n_list = self.classes['root_id']
        nd_met = []
        for i,n in enumerate(npvector):
            if type(n)==str:
                if neuronclass in n:
                    print(n)
                    nd_met = np.append(nd_met,i)
        
        n_number = n_list[nd_met]

        return n_number
    
    def simulation_NP_class(self,neuronclass,act_len,iterations):
        # Stimulate a neuprint neuron class/classes and see what happens
        df_meta = self.df_meta.copy()
        np_type = df_meta['hemibrain_type']
        act_vec = np.zeros([len(np_type),iterations],dtype='float64')
        dx = np.isin(np_type,neuronclass)
        act_vec[dx,:act_len] = 1
        self.act_vec = act_vec
        self.activity_matrix = self.simulation_engine(act_vec)
        
    def simulation_engine(self,act_vec):
        # function performs the simulation
        alen = np.shape(act_vec)
        C_mat = self.C_mat
        activity_matrix = np.zeros([alen[0],alen[1]+1],dtype='float64')
        activity_matrix[:,0] = act_vec[:,0]
        for i in range(alen[1]):
            print('Iteration',i)
            av = act_vec[:,i]
            av[av<0] = 0 #rectify
            av = np.expand_dims(av,1)
            av = np.transpose(av)
            av_s = csr_matrix(av,shape=(1,alen[0]),dtype='float64') #need to make matrix sparse
            am = av_s @ C_mat
            
            activity_matrix[:,i+1] = am.todense()
            if i<alen[1]-1:
                act_vec[:,i+1] = act_vec[:,i+1]+activity_matrix[:,i]
                
        activity_matrix = np.transpose(activity_matrix)
        return activity_matrix
        
#%% Testbed


fw = FW_sim()
# choice = {'Type': 'All'}
# fw.initialise_network(choice)
# choice = {'Type': 'NeuronClass','Neuprint': False,'FwClass': True,'FwSuper': False,'NeuronClass': 'optic_lobes'}
# fw.initialise_network(choice)
choice = {'Type': 'Neuropil','Neuprint': False,'FwClass': False,'FwSuper': False,'NeuronClass': ['FB']}
fw.initialise_network(choice,'small')
fw.simulation_NP_class('FB5AB',5,10)
#%%
import matplotlib.pyplot as plt
plt.imshow(fw.activity_matrix,interpolation='none',aspect='auto',vmin=-0.1,vmax=0.1)

#%%
df = fw.connections.copy()
neurotransmitter_effects = {
    'ACH': 1,   
    'DA': -1,     
    'GABA': -1,  
    'GLUT': -1,  
    'OCT': 1,   
    'SER': 1    
    }
df['nt_sign'] = df['nt_type'].map(neurotransmitter_effects)#convert type to sign in the data frame
df['syn_cnt_sgn'] = df['syn_count']*df['nt_sign']#multiply count by sign for unscaled 'effectome'
df = df.groupby(['pre_root_id', 'post_root_id']).agg({'syn_cnt_sgn': 'sum', 'syn_count': 'sum', 'neuropil':'first', 
       'nt_type':'first'}).reset_index()#sum synapses across all unique pre and post pairs


n_number = fw.n_number.copy()

pre = pd.Series.to_numpy(df['pre_root_id'],dtype='int64')

dx1 = np.isin(pre,n_number)

post = pd.Series.to_numpy(df['post_root_id'],dtype='int64')
dx2 = np.in1d(post,n_number)

dx = dx1&dx2
print(len(np.unique(df['pre_root_id'][dx])))

    
dx_i = [i for i,r in enumerate(dx) if r>0.1]
n_check = np.append(df['pre_root_id'][dx],df['post_root_id'][dx])
n_check = np.unique(n_check)
print(len(n_check))
#%%
