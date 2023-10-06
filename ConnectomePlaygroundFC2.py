# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:45:04 2023

@author: dowel
"""

import ConnectomeFunctions as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, leaves_list
#%% Tangential neuron analysis
from neuprint import fetch_neurons, NeuronCriteria as NC
from neuprint import queries 
#%% 1 Get all TN inputs and outputs. 
# Create array where columns are inputs/outputs and rows are TNs


out_types, in_types, in_array, out_array, Names = cf.input_output_matrix(['FC2.*'])



rowsums = np.nansum(in_array,axis=1)
in_cl = in_array/rowsums[:,np.newaxis]
rowsums = np.sum(out_array,axis=1)
out_cl = out_array/rowsums[:,np.newaxis]

cluster_in,dmat_in = cf.hier_cosine(in_cl,0.7)
cluster_out,dmat_out = cf.hier_cosine(out_cl,0.7)
cluster_in_out,dmat_in_out = cf.hier_cosine(np.append(in_cl,out_cl,1),0.7)
#%%
top_in = np.sum(in_cl,0)>0.01
#top_in = np.sum(top_in,0)>0
plt.imshow(in_cl[:,top_in],vmax=0.05,aspect='auto',interpolation='none',cmap='Greys_r')
plt.yticks(np.linspace(0,len(Names)-1,len(Names)),labels=Names,fontsize=12)
plt.xticks(np.linspace(0,sum(top_in)-1,sum(top_in)),labels=in_types[top_in],rotation=90,fontsize=12)
plt.xlabel('Inputs',fontsize=12)
plt.gcf().subplots_adjust(bottom=0.3)
#%% 
out_types, in_types, in_array, out_array, Names = cf.input_output_matrix(['PFL3','PFL.*'])



rowsums = np.nansum(in_array,axis=1)
in_cl = in_array/rowsums[:,np.newaxis]
rowsums = np.sum(out_array,axis=1)
out_cl = out_array/rowsums[:,np.newaxis]


#%%
top_in = np.sum(in_cl,0)>0.01
#top_in = np.sum(top_in,0)>0
plt.imshow(in_cl[:,top_in],vmax=0.05,aspect='auto',interpolation='none',cmap='Greys_r')
plt.yticks(np.linspace(0,len(Names)-1,len(Names)),labels=Names,fontsize=12)
plt.xticks(np.linspace(0,sum(top_in)-1,sum(top_in)),labels=in_types[top_in],rotation=90,fontsize=12)
plt.xlabel('Inputs',fontsize=12)
plt.gcf().subplots_adjust(bottom=0.3)