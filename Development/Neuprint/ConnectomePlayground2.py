# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:32:27 2023

@author: dowel
"""

#%% Load up modules
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


neuron_df, conn_df = fetch_adjacencies(NC(type='MBON.*'), NC(type='FB.*'))

#%% 
# %%

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
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

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

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
def cluster_heat(model,Tan_Names,d_mat,dthresh=0.5):
    l_list = linkage_order(model)
    dmrank = d_mat[l_list,:]
    dmrank = dmrank[:,l_list]
    tan_index = [int(Tan_Names[i][2]) for i in l_list]
    cmap =  plt.get_cmap('turbo', 9)
    cmap_rgb = np.empty([10,4])
    for i in range(9):
        cmap_rgb[i+1,:] = cmap(i / (8)) 
    clist = cmap_rgb[tan_index]
    plt.Figure()
    ax = plt.subplot()
    plt.imshow(1-dmrank,cmap = 'Greys_r', vmax =  dthresh)

    yt =  np.linspace(0,len(l_list)-1,len(l_list))
    plt.yticks(ticks=yt,labels = Tan_Names[l_list])
    plt.xticks(ticks=yt,labels = Tan_Names[l_list])
    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(clist[i,:])
        plt.show()
        
    for i, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(clist[i,:])
        
    plt.xticks(fontsize = 8,rotation=90)
    
def weight_heat(model,weight_mat,Tan_Names,W_names,dthresh):
    l_list = linkage_order(model)
    weight_rank = weight_mat[l_list,:]
    tan_index = [int(Tan_Names[i][2]) for i in l_list]
    # cmap = np.array([[0, 0, 0],
    #                  [142,1,82],
    # [197,27,125],
    # [222,119,174],
    # [241,182,218],
    # [230,245,208],
    # [184,225,134],
    # [127,188,65],
    # [77,146,33],
    # [39,100,25]])/255
    cmap =  plt.get_cmap('turbo', 9)
    cmap_rgb = np.empty([10,4])
    for i in range(9):
        cmap_rgb[i+1,:] = cmap(i / (8)) 
    clist = cmap_rgb[tan_index,:]
    clist = cmap[tan_index]
    plt.Figure()
    ax = plt.subplot()
    plt.imshow(weight_rank,cmap = 'Greys_r', vmax =  dthresh)
    yt =  np.linspace(0,len(l_list)-1,len(l_list))
    xt = np.linspace(0,len(W_names)-1,len(W_names))
    plt.yticks(ticks=yt,labels = Tan_Names[l_list])
    plt.xticks(ticks=xt,labels = W_names)
    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(clist[i,:])
        plt.show()
        
    plt.xticks(fontsize = 8,rotation=90)

def weight_heat_MBON(model,weight_mat,Tan_Names,W_names,dthresh, MBON_val):
    l_list = linkage_order(model)
    weight_rank = weight_mat[l_list,:]
    tan_index = [int(Tan_Names[i][2]) for i in l_list]
   
    cmap =  plt.get_cmap('turbo', 9)
    cmap_rgb = np.empty([10,4])
    for i in range(9):
        cmap_rgb[i+1,:] = cmap(i / (8)) 
    clist = cmap_rgb[tan_index,:]
    plt.Figure()
    ax = plt.subplot()
    plt.imshow(weight_rank,cmap = 'Greys_r', vmax =  dthresh)
    yt =  np.linspace(0,len(l_list)-1,len(l_list))
    xt = np.linspace(0,len(W_names)-1,len(W_names))
    plt.yticks(ticks=yt,labels = Tan_Names[l_list])
    plt.xticks(ticks=xt,labels = W_names)
    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(clist[i,:])
        plt.show()
    
    cmap = np.array([
                     [142,1,82],
    [39,100,25]])/255
    for i, xtick in enumerate(ax.get_xticklabels()):
        if MBON_val[i]>0:
            xtick.set_color(cmap[0,:])
        else :
            xtick.set_color(cmap[1,:])
    
    plt.xticks(fontsize = 8,rotation=90)

def weight_heat_MBON2(l_list,weight_mat,Tan_Names,W_names,dthresh, MBON_val):
    weight_rank = weight_mat[l_list,:]
    tan_index = [int(Tan_Names[i][2]) for i in l_list]
   
    cmap =  plt.get_cmap('turbo', 9)
    cmap_rgb = np.empty([10,4])
    for i in range(9):
        cmap_rgb[i+1,:] = cmap(i / (8)) 
    clist = cmap_rgb[tan_index,:]
    plt.Figure()
    ax = plt.subplot()
    plt.imshow(weight_rank,cmap = 'Greys_r', vmax =  dthresh)
    yt =  np.linspace(0,len(l_list)-1,len(l_list))
    xt = np.linspace(0,len(W_names)-1,len(W_names))
    plt.yticks(ticks=yt,labels = Tan_Names[l_list])
    plt.xticks(ticks=xt,labels = W_names)
    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(clist[i,:])
        plt.show()
    
    cmap = np.array([
                     [142,1,82],
    [39,100,25]])/255
    for i, xtick in enumerate(ax.get_xticklabels()):
        if MBON_val[i]>0:
            xtick.set_color(cmap[0,:])
        else :
            xtick.set_color(cmap[1,:])
    
    plt.xticks(fontsize = 8,rotation=90)    
    
def weight_heat_FB(model,weight_mat,Tan_Names,W_names,dthresh):
    l_list = linkage_order(model)
    weight_rank = weight_mat[l_list,:]
    tan_index = [int(Tan_Names[i][2]) for i in l_list]
    # cmap = np.array([[0, 0, 0],
    #                  [142,1,82],
    # [197,27,125],
    # [222,119,174],
    # [241,182,218],
    # [230,245,208],
    # [184,225,134],
    # [127,188,65],
    # [77,146,33],
    # [39,100,25]])/255
    cmap =  plt.get_cmap('turbo', 9)
    cmap_rgb = np.empty([10,4])
    for i in range(9):
        cmap_rgb[i+1,:] = cmap(i / (8)) 
    clist = cmap_rgb[tan_index,:]
    plt.Figure()
    ax = plt.subplot()
    plt.imshow(weight_rank,cmap = 'Greys_r', vmax =  dthresh)
    yt =  np.linspace(0,len(l_list)-1,len(l_list))
    xt = np.linspace(0,len(W_names)-1,len(W_names))
    plt.yticks(ticks=yt,labels = Tan_Names[l_list])
    plt.xticks(ticks=xt,labels = W_names)
    for i, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(clist[i,:])
        
    for i, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(clist[i,:])
   
    
    plt.xticks(fontsize = 8,rotation=90)
#%% 2 Do hiearchical clustering of all inputs/outputs

rowsums = np.nansum(in_array,axis=1)
in_cl = in_array/rowsums[:,np.newaxis]
rowsums = np.sum(out_array,axis=1)
out_cl = out_array/rowsums[:,np.newaxis]

cluster_in,dmat_in = hier_cosine(in_cl,0.7)
cluster_out,dmat_out = hier_cosine(out_cl,0.7)
cluster_in_out,dmat_in_out = hier_cosine(np.append(in_cl,out_cl,1),0.7)

tandx = np.empty(np.shape(Tan_Names),dtype='int')
for i,t in enumerate(Tan_Names):

    tandx[i] = np.where(out_types==t)[0]

out_tan = out_cl[:,tandx]
cluster_tan,dmat_tan = hier_cosine(out_tan,0.7)
#%% Tan connectivit
plt.Figure()
plot_dendrogram(cluster_tan,truncate_mode = None, color_threshold = 0.3,labels = Tan_Names)
plt.xticks(fontsize = 10)
#%%
cluster_heat(cluster_tan,Tan_Names,dmat_tan,1)
#%% Input dendrogram
plt.Figure()
plot_dendrogram(cluster_in,truncate_mode = None, color_threshold = 0.3,labels = Tan_Names)
plt.xticks(fontsize = 10)
plt.show()
#%% Input heatmap
cluster_heat(cluster_in,Tan_Names,dmat_in,1)
plt.xticks(fontsize = 6)
plt.yticks(fontsize = 6)
savedir = "C:\\Users\dowel\\Documents\\PostDoc\\ConnectomeMining\\TangentialClustering\\RoughPlots\\"
savename = 'InputClust.png'
savefile = savedir+savename
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(savefile)
plt.show()
#%% Output dendrogram
plt.Figure()
plot_dendrogram(cluster_out,truncate_mode = None, color_threshold = 0.3,labels = Tan_Names)
plt.xticks(fontsize = 10)
plt.show()
#%% Output heatmap
cluster_heat(cluster_out,Tan_Names,dmat_out,1)
savedir = "C:\\Users\dowel\\Documents\\PostDoc\\ConnectomeMining\\TangentialClustering\\RoughPlots\\"
savename = 'OutputClust.png'
savefile = savedir+savename
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(savefile)
plt.show()

#%% Input and output dendrogram
plt.Figure()
plot_dendrogram(cluster_in,truncate_mode = None, color_threshold = 0.7,labels = Tan_Names)
plt.xticks(fontsize = 10)
plt.show()
#%% Input and output heatmap
cluster_heat(cluster_in_out,Tan_Names,dmat_in_out,0.75)

#%% MBON clustering
neuron_type = 'MBON'
mbon_dx = [i for i, s in enumerate(in_types) if neuron_type in s]
mbon_in_All = in_cl[:,np.transpose(mbon_dx,0)]

mbon_names = in_types[np.transpose(mbon_dx,0)]
zdex = np.sum(mbon_in_All,axis=1)<0.00000001
mbon_in = mbon_in_All[~zdex,:]
Tan_Names_mbon = Tan_Names[~zdex]
cluster_mbon,dmat_mbon = hier_cosine(mbon_in,0.4)
#%% Plot dendrogram
plot_dendrogram(cluster_mbon,truncate_mode = None, color_threshold = 0.4,labels = Tan_Names_mbon)
plt.xticks(fontsize = 10)
plt.show()


#%% Get MBON types and valences


mbdict = ["a'1(R)", 
 "a'2(R)", 
 "a'3(R)", 
 "a'L(L)", 
 "a'L(R)", 
 'a1(R)', 
 'a2(R)', 
 'a3(R)', 
 'aL(L)', 
 'aL(R)', 
 "b'1(R)", 
 "b'2(R)", 
 "b'L(L)", 
 "b'L(R)",
 'b1(R)', 
 'b2(R)', 
 'bL(L)', 
 'bL(R)', 
 'g1(R)', 
 'g2(R)', 
 'g3(R)', 
 'g4(R)', 
 'g5(R)', 
 'gL(L)', 
 'gL(R)' ]
mvalence = np.array([ -1 ,
            -1
,-1
, -1
,-1
,1
,-1
, -1
, 0
, 0
,0
, 1
, 1
,  1
, 1
,  1
, 1
,  1
, -1
, -1
,  1
,  1
,  1
,  0 
, 0])
typevalence = np.empty(len(mbon_names))
for i, m in enumerate(mbon_names):
    neurons_df, roi_counts_df = fetch_neurons(NC(type=m))
    ln = len(neurons_df)
    valencies = np.empty(ln)
    for r in range(ln):
        roi_in = neurons_df['inputRois'][r]
        rdx = np.where(np.in1d(mbdict,roi_in))
        rdx = rdx[0]
        rdx = np.transpose(rdx,0)
        valencies[r] = np.sum(mvalence[rdx])
    typevalence[i] = np.sum(valencies)/ln
 #%% 

i = np.argsort(typevalence)[::-1]
tv = typevalence[i]
weight_heat_MBON(cluster_mbon,mbon_in[:,i],Tan_Names_mbon,mbon_names[i],0.05,tv)


#%%
weight_heat_MBON(cluster_out,mbon_in_All[:,i],Tan_Names,mbon_names[i],0.01,tv)

plt.xticks(fontsize = 6)
plt.yticks(fontsize = 6)
savedir = "C:\\Users\dowel\\Documents\\PostDoc\\ConnectomeMining\\TangentialClustering\\RoughPlots\\"
savename = 'OutputClust_MBON_connections.png'
savefile = savedir+savename
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(savefile)
plt.show()
#%% same as above but only showing neurons with mbon>0.01
l_list = linkage_order(cluster_out)
mbon_in_All_s = mbon_in_All[l_list,:]
mbon_in_All_s = mbon_in_All_s[:,i]
Tan_Names_s = Tan_Names[l_list]
mbsum = np.sum(mbon_in_All_s,1)
mbkeep = mbsum>0.01
mbkeep2 = np.where(mbkeep)
mbkeep2 = mbkeep2[0]
weight_heat_MBON2(mbkeep2,mbon_in_All_s,Tan_Names_s,mbon_names[i],0.05,tv)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
savedir = "C:\\Users\dowel\\Documents\\PostDoc\\ConnectomeMining\\TangentialClustering\\RoughPlots\\"
savename = 'OutputClust_MBON_connections_small.png'
savefile = savedir+savename
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(savefile)
plt.show()
#%%
select_outputs_mbon = select_outputs[l_list,:]
tv2 = np.zeros_like(tv)
weight_heat_MBON2(mbkeep2,select_outputs_mbon,Tan_Names_s,select_o_names ,0.05,tv2)

plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
savedir = "C:\\Users\dowel\\Documents\\PostDoc\\ConnectomeMining\\TangentialClustering\\RoughPlots\\"
savename = 'OutputClust_MBON_small_FB_neurons.png'
savefile = savedir+savename
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(savefile)
plt.show()
#%%

weight_heat_MBON(cluster_in,mbon_in_All[:,i],Tan_Names,mbon_names[i],0.01,tv)
#%%
cluster = hier_cosine([in_cl, out_cl],0.7)
plt.Figure()
plot_dendrogram(cluster,truncate_mode = None, color_threshold = 0.7,labels = Tan_Names)
plt.xticks(fontsize = 10)

#%% Look at FC2, HDelta, PFL3 neuron outputs
neuron_type = 'FC2'
FC2_dx = [i for i, s in enumerate(in_types) if neuron_type in s]
FC2_in_All = in_cl[:,np.transpose(FC2_dx,0)]
FC2_names = in_types[np.transpose(FC2_dx,0)]

neuron_type = 'hDelta'
hD_dx = [i for i, s in enumerate(in_types) if neuron_type in s]
hD_in_All = in_cl[:,np.transpose(hD_dx,0)]
hD_names = in_types[np.transpose(hD_dx,0)]

neuron_type = 'vDelta'
vD_dx = [i for i, s in enumerate(in_types) if neuron_type in s]
vD_in_All = in_cl[:,np.transpose(vD_dx,0)]
vD_names = in_types[np.transpose(vD_dx,0)]

neuron_type = 'PFL'
PFL_dx = [i for i, s in enumerate(in_types) if neuron_type in s]
PFL_in_All = in_cl[:,np.transpose(PFL_dx,0)]
PFL_names = in_types[np.transpose(PFL_dx,0)]

select_outputs = np.append(FC2_in_All,hD_in_All,1)
select_outputs = np.append(select_outputs,vD_in_All,1)
select_outputs = np.append(select_outputs,PFL_in_All,1)


select_o_names = np.append(FC2_names,hD_names,0)
select_o_names = np.append(select_o_names,vD_names,0)
select_o_names = np.append(select_o_names,PFL_names,0)
weight_heat_MBON(cluster_out,select_outputs,Tan_Names,select_o_names ,0.01,tv)


savedir = "C:\\Users\dowel\\Documents\\PostDoc\\ConnectomeMining\\TangentialClustering\\RoughPlots\\"
savename = 'OutputClust_SelectOut_connections.png'
plt.xticks(fontsize = 6)
plt.yticks(fontsize = 6)
savefile = savedir+savename
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(savefile)
plt.show()
#%%  Tangential to tangential connections
neuron_type = 'FB'
FB_dx = np.empty(len(Tan_Names))
for i, s in enumerate(Tan_Names):
    dx = np.where(out_types==s)
    FB_dx[i] = dx[0]
FB_dx = list(map(np.int_,FB_dx))
FB_out_All = out_cl[:,FB_dx]

FB_names = in_types[np.transpose(FB_dx,0)]
l_list = linkage_order(cluster_out)
tv = np.ones(len(FB_names))
weight_heat_FB(cluster_out,FB_out_All[:,l_list],Tan_Names,Tan_Names[l_list] ,0.01)

savedir = "C:\\Users\dowel\\Documents\\PostDoc\\ConnectomeMining\\TangentialClustering\\RoughPlots\\"
savename = 'OutputClust_Tan_connections.png'
plt.xticks(fontsize = 6)
plt.yticks(fontsize = 6)
savefile = savedir+savename
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(savefile)
plt.show()
#%% Tangential to tangential clustering
cluster_Tan,dmat_Tan = hier_cosine(FB_out_All,0.7)
plt.Figure()
plot_dendrogram(cluster_Tan,truncate_mode = None, color_threshold = 0.3,labels = Tan_Names)
plt.xticks(fontsize = 10)
plt.show()
#%% Input and output heatmap
cluster_heat(cluster_Tan,Tan_Names,dmat_Tan,1)
#%% 
i = np.argsort(typevalence)[::-1]
tv = typevalence[i]
weight_heat_MBON(cluster_Tan,mbon_in_All[:,i],Tan_Names,mbon_names[i],0.01,tv)

#%%
weight_heat_MBON(cluster_Tan,select_outputs,Tan_Names,select_o_names ,0.01,tv)
#%%
l_list = linkage_order(cluster_Tan)
weight_heat_FB(cluster_Tan,FB_out_All[:,l_list],Tan_Names,Tan_Names[l_list] ,0.01)
#%% Indirect MBON connections
# get full mbon names
criteria = NC(type='MBON.*')
all_mbon = fetch_neurons(criteria)
mbon_names_all = pd.Series.to_numpy(all_mbon[0]['type'])
mbon_names_all = np.unique(mbon_names_all)
tan_secondary = np.zeros([len(Tan_Names),len(mbon_names_all)])

for i, n in enumerate(in_types):
    
    if 'MBON' in n:
        continue
    if 'None' in n:
        continue
    # Get inputs
    try:
        type_in, t_inputs, ncells = cf.top_inputs(n)
    except:
        print('Could not get conns ' + n)
        continue
    # Get MBON proportion
    
    mbondx = [i for i, s in enumerate(type_in) if 'MBON' in s]
    if len(mbondx)==0:
        continue
    
    t_MBONs = type_in[mbondx]
    mbon_w = t_inputs[mbondx]
    weight_norm = np.sum(t_inputs)
    
    # Get outputs
    type_out, t_outputs, ncells = cf.top_inputs(n)
    
    tandx = [i for i, s in enumerate(type_out) if 'FB' in s]
    t_FBs = type_in[tandx]
    print(n)
    for it, t in enumerate(t_FBs):
        
        tdx  = Tan_Names==t
        t_norm = np.sum(in_array[tdx,:])
        
        tin_w = t_outputs[tandx[it]]
        t_in = tin_w/t_norm
        for im,m in enumerate(t_MBONs):
            
            mdx = mbon_names_all==m
            
            mb_in = mbon_w[im]/weight_norm
            tan_secondary[it,mdx] = tan_secondary[it,mdx]+ (mb_in*t_in)
    
    
    
#%% Get type valence

mbdict = ["a'1(R)", 
 "a'2(R)", 
 "a'3(R)", 
 "a'L(L)", 
 "a'L(R)", 
 'a1(R)', 
 'a2(R)', 
 'a3(R)', 
 'aL(L)', 
 'aL(R)', 
 "b'1(R)", 
 "b'2(R)", 
 "b'L(L)", 
 "b'L(R)",
 'b1(R)', 
 'b2(R)', 
 'bL(L)', 
 'bL(R)', 
 'g1(R)', 
 'g2(R)', 
 'g3(R)', 
 'g4(R)', 
 'g5(R)', 
 'gL(L)', 
 'gL(R)' ]
mvalence = np.array([ -1 ,
            -1
,-1
, -1
,-1
,1
,-1
, -1
, 0
, 0
,0
, 1
, 1
,  1
, 1
,  1
, 1
,  1
, -1
, -1
,  1
,  1
,  1
,  0 
, 0])
typevalence = np.empty(len(mbon_names_all))
for i, m in enumerate(mbon_names_all):
    neurons_df, roi_counts_df = fetch_neurons(NC(type=m))
    ln = len(neurons_df)
    valencies = np.empty(ln)
    for r in range(ln):
        roi_in = neurons_df['inputRois'][r]
        rdx = np.where(np.in1d(mbdict,roi_in))
        rdx = rdx[0]
        rdx = np.transpose(rdx,0)
        valencies[r] = np.sum(mvalence[rdx])
    typevalence[i] = np.sum(valencies)/ln
#%% Secondary MBON connections

i = np.argsort(typevalence)[::-1]
tv = typevalence[i]
weight_heat_MBON(cluster_out,tan_secondary[:,i],Tan_Names,mbon_names_all[i],0.005,tv)


#%% 2 Do a 2D embedding with UMAP

# 1. On inputs
rowsums = np.sum(in_array,axis=1)
in_umap = in_array/rowsums[:,np.newaxis]
fit = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='cosine')
u = fit.fit_transform(in_umap)
plt.Figure()
plt.scatter(u[:,0], u[:,1])
#%%
# 2. On outputs
rowsums = np.sum(out_array,axis=1)
out_umap = out_array/rowsums[:,np.newaxis]
fit = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='cosine')
u = fit.fit_transform(in_umap)
plt.Figure()
plt.scatter(u[:,0], u[:,1])
#%% 3. Scatter neurons based along dimensions 


#%% Plot inputs for different neurons - simple visualisation
typelib, t_inputs, ncells = cf.top_inputs('hDeltaC')

x = np.arange(0,len(typelib))
i = np.argsort(t_inputs)[::-1]

t_inputs = t_inputs[i]
typelib = typelib[i]
ncells = ncells[i]
top_i = np.linspace(0,19,20,dtype = 'int')

tan_index = [i for i, s in enumerate(typelib) if "FB" in s]

    
f = plt.Figure(figsize = (6,4))
plt.plot(x[top_i],t_inputs[top_i])
plt.xticks(x[top_i],typelib[top_i],rotation=45,fontsize = 10)
plt.title('hDeltaC inputs')
plt.ion()
plt.show()
#%%
mn_in = t_inputs/ncells
f = plt.Figure(figsize = (6,4))
ymax = np.max(t_inputs)

plt.scatter(mn_in,t_inputs)
plt.scatter(mn_in[tan_index],t_inputs[tan_index],c='g')

plt.plot([0, ymax], [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],2), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],5), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],10), [0, ymax], color = 'k')



plt.xlabel('Mean input')
plt.ylabel('total input')
plt.title('hDeltaC')
plt.show()
del tan_index