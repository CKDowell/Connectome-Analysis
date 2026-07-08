# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:47:25 2026

@author: dowel

Work to try and characterise CRE neurons - terra nova



"""

from Stable.CorrectFlyWire import fw_corrections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#%%
top_dir = 'D:\\ConnectomeData\\FlywireWholeBrain\\'

fn = top_dir + 'connections_princeton_no_threshold.csv'
df = pd.read_csv(fn)
fn = top_dir + 'classification.csv'
df_meta = pd.read_csv(fn)
fn = top_dir +'processed_labels.csv'
df_meta_com = pd.read_csv(fn)


fn = top_dir + 'consolidated_cell_types.csv'
df_meta_typ = pd.read_csv(fn)
df['syn_count_sq'] = df['syn_count']**2
total_inputs = df.groupby('post_pt_root_id')['syn_count_sq'].sum()**.5
total_outputs = df.groupby('pre_pt_root_id')['syn_count_sq'].sum()**.5

df['weight_norm_in'] = df['syn_count'] / df['post_pt_root_id'].map(total_inputs)
df['weight_norm_out'] = df['syn_count'] / df['pre_pt_root_id'].map(total_outputs)

#%% CRE index


credx = [i  for i,t in enumerate(df_meta_typ['primary_type']) if 'SMP' in t]

neurons = df_meta_typ['root_id'][credx]

dx = np.logical_or(np.in1d(df['post_pt_root_id'],neurons),np.in1d(df['pre_pt_root_id'],neurons))

dfsmall = df[dx]

C = dfsmall.pivot_table(index='pre_pt_root_id',columns='post_pt_root_id',values='syn_count',aggfunc='sum',fill_value=0)

predx = np.in1d(C.index,neurons)
postdx = np.in1d(C.columns,neurons)
Cn = C.to_numpy()

Cin =Cn[:,postdx]
Cout = Cn[predx,:]
C_cre = Cin[predx,:]

Cin = Cin/np.sqrt(np.sum(Cin**2,axis=0))
Cout = Cout/np.sqrt(np.sum(Cout**2,axis=1)[:,np.newaxis])

Cin_cre = C_cre/np.sqrt(np.sum(C_cre**2,axis=0))
Cout_cre = C_cre/np.sqrt(np.sum(C_cre**2,axis=1)[:,np.newaxis])

Cosin = Cin.T @ Cin

Cosout = Cout @ Cout.T



Cosincre = Cin_cre.T @ Cin_cre
Cosoutcre = Cout_cre @ Cout_cre.T

#%% Cluster

from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import leaves_list

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


# inputs
d_mat = Cosin-1
distance_thresh = .5
cluster = AgglomerativeClustering(linkage='single', 
                               compute_distances = True, distance_threshold =distance_thresh, n_clusters = None)
cluster.fit(d_mat)


z = linkage_order(cluster)

Cosin2 = Cosin[z,:]
Cosin2 = Cosin2[:,z]
plt.imshow(Cosin2,interpolation='None',vmin=-.5,vmax=.5,cmap='coolwarm')

cneurons = C.columns[postdx]
cell_types = np.empty(len(neurons),dtype='U50')



for i,n in enumerate(cneurons):
    dx = df_meta_typ['root_id']==n
    if np.sum(dx)==0:
        cell_types[i] = 'unassigned'
        
    else:
        cell_types[i] = df_meta_typ['primary_type'][dx].iloc[0]
        
cell_types_inord = cell_types[z]
plt.xticks(np.arange(0,len(cell_types)),labels=cell_types,rotation=90)
plt.yticks(np.arange(0,len(cell_types)),labels=cell_types)

# Cre cre con
credx = np.where(postdx)[0]

csmall = Cin[credx,:]
csmall = csmall[z,:]
csmall = csmall[:,z]
plt.figure()
plt.imshow(csmall,vmin=0,vmax=.01)

# outputs
d_mat = Cosout-1
distance_thresh = .5
cluster = AgglomerativeClustering(linkage='single', 
                               compute_distances = True, distance_threshold =distance_thresh, n_clusters = None)
cluster.fit(d_mat)


z = linkage_order(cluster)

Cosin2 = Cosin[z,:]
Cosin2 = Cosin2[:,z]
plt.figure()
plt.imshow(Cosin2,interpolation='None',vmin=-.5,vmax=.5,cmap='coolwarm')

cneurons = C.index[predx]


cell_types_inord = cell_types[z]
plt.xticks(np.arange(0,len(cell_types)),labels=cell_types,rotation=90)
plt.yticks(np.arange(0,len(cell_types)),labels=cell_types)

# Cre cre con
credx = np.where(postdx)[0]

csmall = Cin[credx,:]
csmall = csmall[z,:]
csmall = csmall[:,z]
plt.figure()
plt.imshow(csmall,vmin=0,vmax=.01)

d_mat = Cosincre-1
distance_thresh = .5
cluster = AgglomerativeClustering(linkage='single', 
                               compute_distances = True, distance_threshold =distance_thresh, n_clusters = None)
cluster.fit(d_mat)


z = linkage_order(cluster)

Cosin2 = Cosincre[z,:]
Cosin2 = Cosincre[:,z]
plt.figure()
plt.imshow(Cosin2,interpolation='None',vmin=-.5,vmax=.5,cmap='coolwarm')
#%%
