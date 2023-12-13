# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:18:46 2023

@author: dowel
"""

#%% Import packages
from Stable.SimulationFunctions import sim_functions as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
# from neuprint import Client
# c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
# c.fetch_version()
from neuprint import fetch_adjacencies, fetch_neurons, NeuronCriteria as NC

#%% 


#%% Simulate columnar input to FB
S = sf()

columnar_neurons = ['hDeltaA','hDeltaB','hDeltaC','hDeltaD','hDeltaE',
                    'hDeltaF','hDeltaG','hDeltaH','hDeltaI','hDeltaJ',
                    'hDeltaK','hDeltaL','hDeltaM','PFNa','PFNv','PFNd',
                    'PFR','PFNm_a','PFNm_b','PFNp_a','PFNp_b','PFNp_c',
                    'PFNp_d','FC1A','FC1B','FC1C','FC1D',
                    'FC1E','FC2A','FC2B','FC2C','FC3','PFL1','PFL2','PFL3']

FB_network = ['FB.*','hDelta.*','FC.*','PFL.*','vDelta.*','PFN.*','PFR.*','FR.*','PFG.*','FS.*','PFL.*']
sim_output = {}
for c in columnar_neurons:
    print(c)
    sim_output[c] = S.run_sim_act([c],FB_network,5)

#%% plot effect on hDelta C
savedir = "Y:\Data\Connectome\Connectome Mining\ActivationSimulations\ColumnarStimulation"
layers = [1,2,3] # Look at each iteration layer
cnt = 0
plt.figure(figsize=(15,5))
xticknames = []
this_neuron = 'FC2B'
for i in layers:
    for c in columnar_neurons:
        if c == this_neuron:
            continue
        tsim = sim_output[c]
        sim_mat = tsim['MeanActivityType']
        t_names = tsim['TypesSmall']
        n_dx = t_names== this_neuron
        ts = sim_mat[:,n_dx]
        plt.scatter(cnt,ts[i],color='k')
        cnt = cnt+1 
        xticknames = np.append(xticknames,c)
    plt.plot([cnt-0.5,cnt-0.5],[-0.03, 0.03],linestyle='--',color='r')
plt.subplots_adjust(bottom=0.3)
plt.xticks(np.linspace(0,cnt-1,cnt),labels=xticknames,rotation=90)
plt.xlabel('Stimulated neurons')
plt.title('Recipient neuron: ' + this_neuron)
plt.ylabel('Activity')
plt.show()


savepath = os.path.join(savedir,"Col_stim"+this_neuron+".png")
plt.savefig(savepath)

#%% hDelta C connectivity matrix
from Stable.ConnectomeFunctions import  input_output_matrix


out_types, in_types, in_array, out_array, types_u = input_output_matrix('PFL3')
in_types = in_types[in_array.flatten()>100]

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
        
dx = np.transpose((-in_array).argsort()[:])
dx = dx.flatten()

# %%

from Stable.ConnectomeFunctions import hier_cosine
dx_rem = in_types=='None'

cm = con_matrix[~dx_rem,:]
cm = cm[:,~dx_rem]
cluster, dmat = hier_cosine(cm,1)
#%% 
from Stable.ConnectomeFunctions import linkage_order
z = linkage_order(cluster)
cm2 = cm[z,:]
cm2 = cm2[:,z]
in_names = in_types[~dx_rem]
in_names = in_names[z]
#%% 
plt.figure(figsize=(20,20))
plt.imshow(cm2,vmin=0,vmax = 500)
in_array_sort = in_names
xt = np.linspace(0,len(in_array_sort)-1,len(in_array_sort))
plt.xticks(xt,labels=in_array_sort,rotation=90,fontsize=10)
plt.yticks(xt,labels=in_array_sort,fontsize=10)
plt.subplots_adjust(bottom=0.2)
plt.show()
#%% Dendrite vs axon location finder
from neuprint import Client
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
from neuprint import NeuronCriteria as NC, SynapseCriteria as SC, fetch_synapses, fetch_synapse_connections
from scipy import stats
from sklearn import mixture
#%% Try for example cell
cell = 879450931
def get_den_ax_loc(cell,num_gauss=3):
    """
    Function determines axon and dendrite location from cell type
    Returns: 
        ax_term: axon terminal mean location
        den_term: mean locations of dendrite terminals
    How it works:
        Function takes the mean pre-synapse location to be the axon terminal
        Then makes a guassian mixture model on post synapse terminals
        Where the two don't overlap are where dendrites are
    Future extensions: 
        Use gaussian mixture modelling to get terminal location for
        cells with multiple axon terminal sites
    Limitations: 
        You should know in advance how the neurite tree looks to specify the
        number of gaussians
    
    """
    
    syndf = fetch_synapses(cell)
    pre_post = syndf['type']
    pre = pd.Series.to_numpy(pre_post=='pre')
    post = pd.Series.to_numpy(pre_post=='post')
    syn_locs = np.array([syndf['x'], syndf['y'], syndf['z']])
    
    cdx = pd.Series.to_numpy(syndf['confidence']>0.9) # Can relax this if needs be
    syn_locs = np.transpose(syn_locs)
    pre_locs = syn_locs[pre&cdx,:]
    post_locs = syn_locs[post&cdx,:]
    
    # get mean pre location
    mn_pre = np.mean(pre_locs,axis=0)
    sd_pre = np.std(pre_locs,axis=0)
    #
    gm = mixture.GaussianMixture(n_components=num_gauss, random_state=0).fit(post_locs)
    # Gaussian mixture model with two gaussians
    mn_post = gm.means_
    
    ax_term = mn_pre
    den_dx = np.sqrt(np.sum(np.square(mn_post-mn_pre),axis=1))
    dx = [True, True, True]
    ax_dx = np.argmin(den_dx)
    dx[ax_dx] = False
    den_term = mn_post[dx,:]
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(post_locs[:,0],post_locs[:,1],post_locs[:,2],color='k')
    # ax.scatter(pre_locs[:,0],pre_locs[:,1],pre_locs[:,2],color='r')
    # ax.scatter(mn_post[:,0],mn_post[:,1],mn_post[:,2],color='b')
    # plt.show()
    
    return ax_term, den_term
get_den_ax_loc(cell)
#%% Specific inputs onto axons vs dendrites

def get_syn_pair_loc(cell_pre,cell_post):
    # First get synapse information of post-synaptic neuron
    ax_term, den_term = get_den_ax_loc(cell_post,num_gauss=3)
    # Get synapse information about the pair
    df_syn = fetch_synapse_connections(cell_pre,cell_post)
    syn_post_big = np.array([df_syn['x_post'],df_syn['y_post'],df_syn['z_post']]).transpose()
    syn_loc= {}
    for i_n,n in enumerate(cell_pre):
        dx = pd.Series.to_numpy(df_syn['bodyId_pre']==n)
        syn_post = syn_post_big[dx,:]
        dshape = np.shape(den_term)
        syn_shape = np.shape(syn_post)
        ax_den = np.empty([syn_shape[0],dshape[0]+1],dtype=float)
        ax_diff = np.sqrt(np.sum(np.square(syn_post-ax_term),axis=1))
        ax_den[:,0] = ax_diff
        for i in range(dshape[0]):
            d_diff = np.sqrt(np.sum(np.square(syn_post-den_term[0,:]),axis=1)) 
            ax_den[:,i+1] = d_diff
            
        d_code = np.argmin(ax_den,axis=1)
        ax = d_code==0    
        syn_loc[i_n] = {'ax_den': ax_den, 'd_code': d_code, 'is_axon': ax}    
    return syn_loc,ax_term,den_term 

cell_post = 949054716
cell_pre = [977002744,1165314688]
syn_loc,ax_term,den_term = get_syn_pair_loc(cell_pre,cell_post)
        
#%% Axon dendrite h_delta

def ax_dendrite_hDelta(hdelta):
    """
        Parameters
        ----------
        hdelta : 
        Type of hDelta neuron
       
        Returns
        Axon vs denrite location of all pre synapses by type
        -------
        None.
       
    """   
    # Get all hdeltaC type ids
    criteria = NC(type=hdelta)
    n_df1,ndf2 = fetch_neurons(criteria)
    post_ids = n_df1['bodyId']
    # Get a summary of inputs to create the matrix
    neuron_df, conn_df, =  fetch_adjacencies(None,criteria)
    uni_type = np.unique(pd.Series.to_numpy(neuron_df['type'],'str'))
    
    ax_den_mat = np.zeros([len(uni_type),2],dtype=float)
    ax_den_mat_all = np.zeros([len(uni_type),3,len(post_ids)],dtype=float)
    ax_den_mat_adv = np.zeros([len(uni_type),3],dtype=float)
    
    # Iterate through each hdelta C neuron
    for i, n in enumerate(post_ids):
        print(str(i),  ' of ' ,str(len(post_ids)))
        dx = conn_df['bodyId_post']==n
        cell_pre = pd.Series.to_numpy(conn_df['bodyId_pre'][dx],dtype='int64')
        syn_loc,ax_term,den_term = get_syn_pair_loc(cell_pre,n)
        v = np.argmin(den_term[:,2])
        d = np.argmax(den_term[:,2])
        for i_s in range(len(cell_pre)):
            print(i_s)
            s = syn_loc[i_s]
            axnum = np.sum(s['d_code']==0)
            denum = np.sum(s['d_code']!=0)
            
            de_d = np.sum(s['d_code']==d)
            de_v = np.sum(s['d_code']==v)
            cdx = neuron_df['bodyId']==cell_pre[i_s]
            tp = pd.Series.to_numpy(neuron_df['type'][cdx])
            tdx = uni_type ==tp
            ax_den_mat[tdx,0] = ax_den_mat[tdx,0]+axnum 
            ax_den_mat[tdx,1] = ax_den_mat[tdx,1]+denum 
            ax_den_mat_adv[tdx,0] = ax_den_mat_adv[tdx,0]+axnum 
            ax_den_mat_adv[tdx,1] = ax_den_mat_adv[tdx,1]+de_d 
            ax_den_mat_adv[tdx,2] = ax_den_mat_adv[tdx,2]+de_v 
            
            ax_den_mat_all[tdx,0,i] = ax_den_mat_all[tdx,0,i]+axnum 
            ax_den_mat_all[tdx,1,i] = ax_den_mat_all[tdx,1,i]+de_d 
            ax_den_mat_all[tdx,2,i] = ax_den_mat_all[tdx,2,i]+de_v
            
    h_delta_ax_den = {'AxonDendrite': ax_den_mat,'AxonDendriteAdv': ax_den_mat_adv,
                      'AxonDendriteAll': ax_den_mat_all,'Post_types': uni_type,'hDelta': hdelta}
    return h_delta_ax_den



#%% 
plt.close('all')
hdeltas = ['hDeltaA', 'hDeltaB', 'hDeltaC', 'hDeltaD', 'hDeltaE', 'hDeltaF', 
           'hDeltaG', 'hDeltaH', 'hDeltaI', 'hDeltaJ', 'hDeltaK', 'hDeltaL',
           'hDeltaM']
savedir = "Y:\Data\Connectome\Connectome Mining\hDeltaIN_OUT"
for h in hdeltas:
    
    h_delta_ax_den = ax_dendrite_hDelta(h)
    plt_mat = h_delta_ax_den['AxonDendrite'].copy()
    plt_mat[:,0] = plt_mat[:,0]/np.sum(h_delta_ax_den['AxonDendrite'],axis= 1)
    plt_mat[:,1] = plt_mat[:,1]/np.sum(h_delta_ax_den['AxonDendrite'],axis= 1)
    I = np.argsort(plt_mat[:,0])
    plt_mat = plt_mat[I,:]
    mndx = np.sum(h_delta_ax_den['AxonDendrite'][I],axis= 1)>100
    ylab = h_delta_ax_den['Post_types'][I]
    ylab = ylab[mndx]
    
    
    
    
    plt.figure(figsize=(5,20))
    plt.imshow(plt_mat[mndx,:],vmin=0,vmax=1,interpolation='None',aspect='auto')
    
    plt.yticks(np.linspace(0,len(ylab)-1,len(ylab)),labels=ylab)
    plt.ylabel('Input Neuron')
    plt.title(h)
    plt.xticks([0,1],labels =['Axon','Dendrite'])
    plt.show()
    savename = os.path.join(savedir,'DenAx_'+ h +'.png')
    plt.savefig(savename)
    savename = os.path.join(savedir,'DenAx_'+ h +'.pdf')
    plt.rcParams['pdf.fonttype'] = 42 
    plt.savefig(savename)