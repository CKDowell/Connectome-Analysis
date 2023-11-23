# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:24:21 2023

@author: dowel
"""

#%% Plan

# 1) Recapitulate the eigen experiment of the Pillow lab
# 2) Find eigen vectors with neurons of interest - FSB, MBON etc
# 3) Look at angles between eigenvectors to predict how activity spreads/interacts
# 4) Compare these eigenvector interactions to simulations of activity

#%% 

import numpy as np
import pandas as pd
import csv
import os
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
#%% From https://github.com/dp4846/effectome/tree/main
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import scipy as sp

neurotransmitter_effects = {
    'ACH': 1,   
    'DA': -1,     
    'GABA': -1,  
    'GLUT': -1,  
    'OCT': 1,   
    'SER': 1    
    }

# load up dynamics matrix and meta data (get this from https://codex.flywire.ai/api/download)
top_dir = 'D:\\ConnectomeData\\FlywireWholeBrain\\'
fn = top_dir + 'connections.csv'
df = pd.read_csv(fn)#read in
fn = top_dir + 'classification.csv'
df_meta = pd.read_csv(fn)#read in

df['nt_sign'] = df['nt_type'].map(neurotransmitter_effects)#convert type to sign in the data frame
df['syn_cnt_sgn'] = df['syn_count']*df['nt_sign']#multiply count by sign for unscaled 'effectome'
df = df.groupby(['pre_root_id', 'post_root_id']).agg({'syn_cnt_sgn': 'sum', 'syn_count': 'sum', 'neuropil':'first', 
                                                            'nt_type':'first'}).reset_index()#sum synapses across all unique pre and post pairs
vals, inds, inv = np.unique(list(df['pre_root_id'].values) + list(df['post_root_id'].values), return_index=True, return_inverse=True)
conv_dict = {val:i for i, val in enumerate(vals)}
df['pre_root_id_unique'] = [conv_dict[val] for val in df['pre_root_id']]
df['post_root_id_unique'] = [conv_dict[val] for val in df['post_root_id']]
n = len(vals)#total rows of full dynamics matrix
syn_count = df['syn_count'].values
is_syn = (syn_count>0).astype('int')
syn_count_sgn = df['syn_cnt_sgn'].values
pre_root_id_unique = df['pre_root_id_unique'].values
post_root_id_unique = df['post_root_id_unique'].values
# form unscaled sparse matrix
k_eig_vecs = 1000#number of eigenvectors to use
C_orig = csr_matrix((syn_count_sgn, (post_root_id_unique, pre_root_id_unique)), shape=(n, n), dtype='float64')
eigenvalues, eig_vec = eigs(C_orig, k=k_eig_vecs)#get eigenvectors and values (only need first for scaling)
scale_orig = 0.99/np.abs(eigenvalues[0])#make just below stability
W_full = C_orig*scale_orig#scale connectome by largest eigenvalue so that it decays
# dictionary to go back to original ids of cells
conv_dict_rev = {v:k for k, v in conv_dict.items()}
root_id = [conv_dict_rev[i] for i in range(n)]
df_meta_W = df_meta.set_index('root_id', ).loc[root_id]
df_meta_W['root_id_W'] = np.arange(n)
df_meta_W = df_meta_W.set_index('root_id_W')
df_meta_W['root_id'] = root_id

df_sgn = pd.DataFrame({'syn_count_sgn':syn_count_sgn, 'pre_root_id_unique':pre_root_id_unique, 'post_root_id_unique':post_root_id_unique})
df_sgn.to_csv(top_dir + 'connectome_sgn_cnt.csv')
sp.sparse.save_npz(top_dir + 'connectome_sgn_cnt.npz', C_orig)

np.save(top_dir + 'eigenvalues_' + str(k_eig_vecs) + '.npy', eigenvalues)
np.save(top_dir + 'eigvec_' + str(k_eig_vecs) + '.npy', eig_vec)

df_meta_W.to_csv(top_dir + 'meta_data.csv')

df_conv = pd.DataFrame.from_dict(conv_dict_rev, orient='index')
df_conv.to_csv(top_dir + 'C_index_to_rootid.csv')

#%% Check of the 1000 top eigenvectors, which ones have CX neurons
# result is that most do!
#plt.plot(np.abs(eig_vec[:,10]))
has_CX = np.zeros(1000,dtype=int)
for i in range(1000):
    t_eig = eig_vec[:,i]
    dx_srt = np.argsort(np.abs(t_eig))[::-1]
    t_eig_s = np.abs(t_eig[dx_srt])
    te_cs = np.cumsum(t_eig_s)
    te_cs = te_cs/te_cs[-1]
    tdx = te_cs<0.75
    t_sig = dx_srt[tdx]
    
    #t_sig = np.abs(t_eig)>0.02
    t_class = df_meta_W['class'][t_sig]
    cx_dx = t_class=='CX'
    if np.sum(cx_dx)>10:
        has_CX[i] = i
cx_dx = pd.Series.to_numpy(df_meta_W['class']=='CX')
has_CX = has_CX[has_CX>0]
# Output matrix with vector of neurons
plot_matrix = eig_vec[:,has_CX]
dx_plt = np.max(plot_matrix,axis=1)>0.05
plt.imshow(np.transpose(np.real(plot_matrix[dx_plt,:])),vmax=0.3,vmin=0,aspect='auto',interpolation='none')

    

# %% correlation between eigenvectors
eig_vecs_real = []
j=0
while j < len(eig_vec[0]):
    ev = eig_vec[:, j]
    j+=1
    if np.sum(np.imag(ev)**2)>0:
        j+=1
        eig_vecs_real.append(np.imag(ev))
        eig_vecs_real.append(np.real(ev))
    else:
        eig_vecs_real.append(np.real(ev))

eig_vecs_real = np.array(eig_vecs_real)

sort_ind = np.arange(len(eig_vecs_real))
eig_vecs_real = eig_vecs_real/np.sum((eig_vecs_real)**2, 1, keepdims=True)**0.5
eig_cov = eig_vecs_real[sort_ind] @ eig_vecs_real[sort_ind].T

#%% Perform eigen decomposition just on neurons in the central complex - or is that a bad idea
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import numpy as np
import scipy as sp

neurotransmitter_effects = {
    'ACH': 1,   
    'DA': -1,     
    'GABA': -1,  
    'GLUT': -1,  
    'OCT': 1,   
    'SER': 1    
    }

# load up dynamics matrix and meta data (get this from https://codex.flywire.ai/api/download)
top_dir = 'D:\\ConnectomeData\\FlywireWholeBrain\\'
fn = top_dir + 'connections.csv'
df = pd.read_csv(fn)#read in
fn = top_dir + 'classification.csv'
df_meta = pd.read_csv(fn)#read in

df['nt_sign'] = df['nt_type'].map(neurotransmitter_effects)#convert type to sign in the data frame
df['syn_cnt_sgn'] = df['syn_count']*df['nt_sign']#multiply count by sign for unscaled 'effectome'
df = df.groupby(['pre_root_id', 'post_root_id']).agg({'syn_cnt_sgn': 'sum', 'syn_count': 'sum', 'neuropil':'first', 
       'nt_type':'first'}).reset_index()#sum synapses across all unique pre and post pairs

# whittle down df to just neurons with synapses in the fan shaped body
rois = df['neuropil']
r_idx = [i for i,r in enumerate(rois) if r=='FB']
all_nds = np.append(df['pre_root_id'][r_idx],df['post_root_id'][r_idx])
all_nds = np.unique(all_nds)
dx = np.isin(df['pre_root_id'],all_nds) + np.isin(df['post_root_id'],all_nds)
dx = dx>0
dx_i = [i for i,r in enumerate(dx) if r>0]

# any neuron with syn in FB df_small = df.iloc[dx_i]
df_small = df.iloc[r_idx]

vals, inds, inv = np.unique(list(df_small['pre_root_id'].values) + list(df_small['post_root_id'].values), return_index=True, return_inverse=True)
conv_dict = {val:i for i, val in enumerate(vals)}
df_small['pre_root_id_unique'] = [conv_dict[val] for val in df_small['pre_root_id']]
df_small['post_root_id_unique'] = [conv_dict[val] for val in df_small['post_root_id']]
n = len(vals)#total rows of full dynamics matrix
syn_count = df_small['syn_count'].values
is_syn = (syn_count>0).astype('int')
syn_count_sgn = df_small['syn_cnt_sgn'].values
pre_root_id_unique = df_small['pre_root_id_unique'].values
post_root_id_unique = df_small['post_root_id_unique'].values
metdx = np.isin(df_meta['root_id'],vals)
df_meta = df_meta.iloc[metdx]
# form unscaled sparse matrix
k_eig_vecs = 1000#number of eigenvectors to use
C_orig = csr_matrix((syn_count_sgn, (post_root_id_unique, pre_root_id_unique)), shape=(n, n), dtype='float64')
eigenvalues, eig_vec = eigs(C_orig, k=k_eig_vecs)


# dictionary to go back to original ids of cells
conv_dict_rev = {v:k for k, v in conv_dict.items()}
root_id = [conv_dict_rev[i] for i in range(n)]
df_meta_W = df_meta.set_index('root_id', ).loc[root_id]
df_meta_W['root_id_W'] = np.arange(n)
df_meta_W = df_meta_W.set_index('root_id_W')
df_meta_W['root_id'] = root_id
# %% Vectors with known neurons in - test it out

# Start with FB5AB 
neuron_type = 'FB5AB'
nd_met = df_meta_W['hemibrain_type']==neuron_type
ts = df_meta_W['hemibrain_type']
nd_met = []
it = 0
for t in ts:
    if type(t)==str:
        if neuron_type in t:
            print(t)
            nd_met = np.append(nd_met,it)
    it =it+1
    

e_load = np.abs(eig_vec[nd_met.astype(int),:])
#plt.plot(np.transpose(e_load))

i = np.argsort(np.mean(e_load,axis=0))[::-1]

plt.plot(abs(eig_vec[:,i[0]]))
imp_neur = np.abs(eig_vec[:,i[0]])>0.05
i_n = [i for i,r in enumerate(imp_neur) if r>0]

#
plt.imshow(np.abs(eig_vec[:,i[0:49]]),aspect='auto',interpolation='none',vmin = 0, vmax=0.1)
plt.yticks(i_n,labels = df_meta_W['hemibrain_type'][i_n],rotation=0)
corvec = np.zeros(49)
for ix in range(len(corvec)):
    corvec[ix] = eig_cov[i[ix],i[ix+1]]
plt.plot(corvec)
#%% Covariance of eigenv
eig_vecs_real = []
j=0
while j < len(eig_vec[0]):
    ev = eig_vec[:, j]
    j+=1
    if np.sum(np.imag(ev)**2)>0:
        j+=1
        eig_vecs_real.append(np.imag(ev))
        eig_vecs_real.append(np.real(ev))
    else:
        eig_vecs_real.append(np.real(ev))

eig_vecs_real = np.array(eig_vecs_real)

sort_ind = np.arange(len(eig_vecs_real))
eig_vecs_real = eig_vecs_real/np.sum((eig_vecs_real)**2, 1, keepdims=True)**0.5
eig_cov = eig_vecs_real[sort_ind] @ eig_vecs_real[sort_ind].T
#%%
evalu, evec= eigs(eig_cov, k=len(eig_cov))
#%%
datafolder = 'D:\\ConnectomeData\\FlywireWholeBrain'

class fw:
    def __init__(self):
        self.datafolder = 'D:\\ConnectomeData\\FlywireWholeBrain'
    
    def make_conmatrix(self):
        # Function takes the connectivity table and outputs a connectivity matrix
        # with synapse signs
    
    def eigen_domposition(self):
        # Function does eigen decomposition of connectivity matrix
    
    def neuprint_to_FW(self,NPneurons):
        # Function gives flywire names to neuprint neuron types
        
    def eigen_list_NP(self,NPneurons):
        # Function gives the list of egeinvectors for which a given neuron contributes from neuprint name
        
    
    
