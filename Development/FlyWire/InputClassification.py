# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:25:44 2026

@author: dowel

Idea of this script is to take flywire data and classify inputs based upon their
connectivity.

Parameters needed:
    1. Seed neuron type
    2. Search depth (do bear in mind that depths>7 will probably include the whole brain)




"""

#%%
from Stable.CorrectFlyWire import fw_corrections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#%% Sort out type ambiguities if needed %%%%%%%%
#%% Initialise
fw = fw_corrections(classtype='original')
# %% Initialise and get predictions
NP_neuron = 'FB4R'
FW_neuron = 'FB4R'

predictions = fw.allocate_by_connections(NP_neuron,FW_neuron)   

# plot predictions
pcorr = predictions['Corr_pearson']
pcorr_top = predictions['top_pearson']
plt.scatter(pcorr[:,0],pcorr[:,1],color='k')
plt.scatter(pcorr_top[:,0],pcorr_top[:,1],color='r')
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.xlabel('Input Pearson correlation')
plt.ylabel('Output Pearson correlation')
plt.title('Neuron: ' + NP_neuron)
#%% Assign neuron name
ass_thresh = 1.6
pred_ids = predictions['top_candidates']
pred_met = predictions['top_corr_metric']
keep = pred_met>ass_thresh
assign_ids = pred_ids[keep]
fw.update_class(assign_ids,NP_neuron)
#%% 
neurons = predictions['top_candidates']
neurons = [720575940602747360,720575940623310780,720575940622683491,720575940619293622,720575940612972693,720575940626429383,720575940630556743,720575940602920364,720575940622438634,720575940630892344,720575940621838826,720575940607309577,720575940616330267,720575940618744047,720575940639596507,720575940606683228,720575940624103399,720575940630020409,720575940617949861,720575940623590221,720575940623222542,720575940630692143,720575940639598963,720575940636702174,720575940624940336,720575940645448756,720575940615193138,720575940639468789,720575940650870774,720575940617073876]
neurons = [720575940631725011,720575940617494232,720575940635678942,720575940614966214,720575940628907524,720575940614154786,720575940622811116,720575940612851367,720575940620058537,720575940631983277,720575940618694638,720575940623781268,720575940643703588,720575940628219479,720575940629468739,720575940638250112,720575940621347583,720575940631366483,720575940628211244,720575940624215245,720575940632052010]
neurons = [720575940620028513,720575940605379506,720575940639404416,720575940628003739,720575940610894450,720575940640228661,720575940604019424,720575940619237985,720575940630444625,720575940623280457,720575940636069348,720575940632651219]
searchdepth = 3 # number of synapse hops


top_dir = 'D:\\ConnectomeData\\FlywireWholeBrain\\'

fn = top_dir + 'connections_princeton_no_threshold.csv'
df = pd.read_csv(fn)
fn = top_dir + 'classification.csv'
df_meta = pd.read_csv(fn)
fn = top_dir +'processed_labels.csv'
df_meta_com = pd.read_csv(fn)


fn = top_dir + 'consolidated_cell_types.csv'
df_meta_typ = pd.read_csv(fn)

total_inputs = df.groupby('post_pt_root_id')['syn_count'].sum()
df['weight_norm'] = df['syn_count'] / df['post_pt_root_id'].map(total_inputs)




#%% Gather neurons
total_neurons = np.zeros((len(neurons),2),dtype='int64')
total_neurons[:,0] = neurons
search_neurons = neurons.copy()
syn_threshold = 5
df2 = df[df['syn_count']>=syn_threshold]

for i in range(searchdepth):
    dx = np.in1d(df2['post_pt_root_id'],search_neurons)
    pre_neurons = df2['pre_pt_root_id'][dx].unique()
    search_neurons = pre_neurons[~np.in1d(pre_neurons,total_neurons[:,0])][:,np.newaxis]
    add_array = np.append(search_neurons,np.zeros_like(search_neurons)+i+1,axis=1)
    total_neurons = np.append(total_neurons,add_array,axis=0)
#%% Gather neuron names
cell_types = np.empty(len(total_neurons),dtype='U50')

for i,n in enumerate(total_neurons):
    dx = df_meta_typ['root_id']==n[0]
    if np.sum(dx)==0:
        cell_types[i] = 'unassigned'
        
    else:
        cell_types[i] = df_meta_typ['primary_type'][dx].iloc[0]

#%% Make connectivity matrix
from scipy.sparse import coo_matrix, csr_matrix
ndx_pre = np.in1d(df2['pre_pt_root_id'],total_neurons)
ndx_post = np.in1d(df2['post_pt_root_id'],total_neurons)

ndx = np.logical_and(ndx_pre,ndx_post)

df_small = df2[ndx]

C_unsigned = df_small.pivot_table(index='pre_pt_root_id',columns='post_pt_root_id',values='weight_norm',aggfunc='sum',fill_value=0)

C_unsigned = C_unsigned.reindex(index=total_neurons[:,0].squeeze(),columns=total_neurons[:,0].squeeze(),fill_value=0)

C_uns = C_unsigned.to_numpy()
ncount = len(C_uns)

influence_mat = np.zeros((ncount,searchdepth))

ondx = np.where(total_neurons[:,1]==0)[0]

#%% Sparse matrix for operations


#%%


Csparse = csr_matrix(C_uns)
input_sparse = csr_matrix(np.eye(ncount))
for i2 in range(searchdepth):
    out_mat = input_sparse @ Csparse
    influence_mat[:,i2] = np.sum(out_mat[:,ondx],axis=1).squeeze()
    input_sparse = out_mat
    
    
itab = pd.DataFrame({
    "root_id": total_neurons[:,0],
    "depth": total_neurons[:,1],
    "type": cell_types,
    "inf_1": influence_mat[:, 0],
    "inf_2": influence_mat[:, 1],
    "inf_3": influence_mat[:, 2],
})



itabg1 = itab.groupby('type')['inf_1'].sum()
itabg2 = itab.groupby('type')['inf_2'].sum()
itabg3 = itab.groupby('type')['inf_3'].sum()

plt.figure()
plt.plot(itabg1[itabg1>0])
plt.subplots_adjust(bottom=.4)
plt.xticks(rotation=90)
plt.show()


plt.figure()
plt.plot(itabg2[itabg2>0.005])
plt.subplots_adjust(bottom=.4)
plt.xticks(rotation=90)
plt.show()

plt.figure()
plt.plot(itabg3[itabg3>0.005])
plt.subplots_adjust(bottom=.4)
plt.xticks(rotation=90)
plt.show()















    
    
    
    
    
    
    