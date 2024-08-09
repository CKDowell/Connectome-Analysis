# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:06:23 2024

@author: dowel

Aim of this script is to characterise tangential neurons by their connectivity
motifs. This will then inform hypotheses as to how they may function

Expected motifs:
    - Columnar feedback
    - Columnar feedforward
    - Tangential feedbacl
    - Tangential feedforward
    - Output neuron modulation

How to go about it:
    - Identify main outputs of each neuron
    - Identify main inputs of each neuron
    - How many inputs from output neurons
    - How many outputs 2 syns from input - i.e. tan - neuron x - neuron y - tan
    
Practically
    - We have the connectivity matrix
    - For each tan we consider 3 synaptic jumps
    - Gives us a 3 row matrix


"""
import Stable.ConnectomeFunctions as cf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, leaves_list
import os
plt.rcParams['pdf.fonttype'] = 42 
#%% Tangential neuron analysis
from neuprint import fetch_neurons, NeuronCriteria as NC
from neuprint import queries 
#%%
out_types, in_types, in_array, out_array, Names = cf.input_output_matrix(['FB.*'])
#%% get rid of nones
none_dx = in_types=='None'
in_types = in_types[~none_dx]
in_array = in_array[:,~none_dx]

none_dx = out_types=='None'
out_types = out_types[~none_dx]
out_array = out_array[:,~none_dx]

#%%
rowsums = np.nansum(in_array,axis=1)


in_cl = in_array/rowsums[:,np.newaxis]
rowsums = np.sum(out_array,axis=1)


out_cl = out_array/rowsums[:,np.newaxis]
in_cl[np.isnan(in_cl)] = 0

cluster_in,dmat_in = cf.hier_cosine(in_cl,0.7)
cluster_out,dmat_out = cf.hier_cosine(out_cl,0.7)
cluster_in_out,dmat_in_out = cf.hier_cosine(np.append(in_cl,out_cl,1),0.7)
#%% 

z_in = cf.linkage_order(cluster_in)

#%%
top_in = np.sum(out_cl,0)>0.01
#top_in = np.sum(top_in,0)>0
in_im = out_cl[:,top_in]
in_im = in_im[z_in,:]

plt.imshow(in_im,vmax=0.05,aspect='auto',interpolation='none',cmap='Greys_r')
plt.yticks(np.linspace(0,len(Names)-1,len(Names)),labels=Names[z_in],fontsize=5)
plt.xticks(np.linspace(0,sum(top_in)-1,sum(top_in)),labels=out_types[top_in],rotation=90,fontsize=5)
plt.xlabel('Outputs',fontsize=12)
plt.gcf().subplots_adjust(bottom=0.2)
#%%
z_out = cf.linkage_order(cluster_out)

#%% Defined in out
out_dict  = cf.defined_in_out(['FB.*','hDelta.*','FC.*','PF.*','vDelta.*'],['FB.*','hDelta.*','FC.*','PF.*','vDelta.*'])
#out_dict = cf.defined_in_out(['pC.*'],['pC.*'])
#%%
cm = out_dict['con_mat']
plt.imshow(cm,vmax=500,aspect='auto',interpolation='none')
ty = out_dict['in_types']
tyo = out_dict['out_types']
plt.xticks(np.arange(0,len(ty)),labels=ty,rotation=90,fontsize=8)

plt.yticks(np.arange(0,len(ty)),labels=tyo,fontsize=8)
#%%
savedir= "Y:\\Data\\Connectome\\Connectome Mining\\TangentialClustering\\July24"
cluster_cm,dm_cm = cf.hier_cosine(cm,0.7)
z_cm = cf.linkage_order(cluster_cm)
plot_cm = cm[z_cm,:]
plot_cm = plot_cm[:,z_cm]

ty = out_dict['in_types'][z_cm]
tyo = out_dict['out_types'][z_cm]

plt.figure(figsize=(15,15))
plt.imshow(plot_cm,vmax=100,vmin=0,interpolation='none')
plt.xticks(np.arange(0,len(ty)),labels=ty,rotation=90,fontsize=5)
ax = plt.subplot()

FBlist = ['FB6P','FB5J','FB6H','FB5H','FB4M','FB4L','FB4R','FB4X','FB4C','FB5I','FB5AB','FC2A','FC2C','FC2B','PFL3','hDeltaC','hDeltaB','hDeltaJ','FB4P_b']
L = ax.get_xticklabels()
tdx = [i for i,it in enumerate(ty) if it in FBlist]
for i,xtick in enumerate(L):
    tname = ty[i]
    if i in tdx:
        xtick.set_color('r')
    elif 'hDelta' in ty[i]:
        xtick.set_color('b')
plt.yticks(np.arange(0,len(ty)),labels=ty,fontsize=5)
plt.gcf().subplots_adjust(bottom=0.2) 

tdx = [i for i,it in enumerate(ty) if it in FBlist]
for i,xtick in enumerate(ax.get_yticklabels()):
    tname = ty[i]
    if i in tdx:
        xtick.set_color('r')
    elif 'hDelta' in ty[i]:
        xtick.set_color('b')

plt.plot([0,len(ty)],[0,len(ty)],color='w',linestyle='--')
plt.xlim([-0.5,len(ty)-0.5])
plt.ylim([-0.5,len(ty)])
plt.colorbar()
#plt.savefig(os.path.join(savedir,'FSB_HierOrder.pdf'))
#%% save data
import pickle
save_dict = {'con_matrix':cm,'order':z_cm,'data_all':out_dict}
savename = "Y:\\Data\\Connectome\\Hackathon\\connectivity_FSB.pkl"
with open(savename,'wb') as fp:
    pickle.dump(save_dict,fp)
#%%
def x_tick_colours(xt,xtlabs,types,typecolours,**kwargs):
    plt.xticks(xt,labels=xtlabs,**kwargs)
    ax = plt.subplot()
    clist = np.zeros([len(xt),3])
    for i, t in enumerate(types):
        tdx = [i for i, it in enumerate(xtlabs) if t in it]
        clist[tdx,:] = typecolours[i,:]
        
    for i, xtick in enumerate(ax.get_xticklabels()):
        xtick.set_color(clist[i,:])
#%% 



def search_fun(start_id,end_id,cm,count=0):
   # print('search')
    t_row = cm[start_id,:]
    t_id = t_row[end_id]
    if t_id<1 and count<4:
        id_thresh = np.where(t_row>0)[0]
        counts = np.array(np.zeros(len(id_thresh)))-1
        for i,t in enumerate(id_thresh):
            counts[i] = search_fun(t,end_id,cm,count+1)
        count = np.min(counts)
    else:
        count = count+1
        
   # print(count)
    return count

#158
c = search_fun(55,158,cmthresh,0)

#%% Path length matrix
# Go through connectivity matrix and output min path length for neighbours
# See how this compares to con matrix
pthresh = 25
cm = out_dict['con_mat']

cmthresh = (cm>pthresh).astype('int')
cm_min = cmthresh.copy()
types = out_dict['in_types']
for i,n in enumerate(types):
    print(n)
    for i2,n2 in enumerate(types):
        ti = cmthresh[i,i2]
        if ti<1:
            cm_min[i,i2] = search_fun(i,i2,cmthresh)
    
    
    
    
#%%
plt.figure()

cluster_cmr,dmat_cmr = cf.hier_cosine((cm_min-5)<-3,0.7)
z_cmr = cf.linkage_order(cluster_cmr)

cm_minplot = cm_min[z_cmr,:]
cm_minplot =-cm_minplot[:,z_cmr]
cm_minplot[cm_minplot<-3] = -3
ty = out_dict['in_types'][z_cmr]
plt.imshow(cm_minplot,interpolation='none')
plt.xticks(np.arange(0,len(ty)),labels=ty,rotation=90,fontsize=8)
ax = plt.subplot()

FBlist = ['FB6P','FB5J','FB6H','FB5H','FB4M','FB4L','FB4R','FB4X','FB4C','FB5I','FB5AB','FC2A','FC2C','FC2B','PFL3','hDeltaC','hDeltaB','hDeltaJ','FB4P_b']
L = ax.get_xticklabels()
tdx = [i for i,it in enumerate(ty) if it in FBlist]
for i,xtick in enumerate(L):
    if i in tdx:
        xtick.set_color('r')
plt.yticks(np.arange(0,len(ty)),labels=ty,fontsize=6)
plt.gcf().subplots_adjust(bottom=0.2) 

tdx = [i for i,it in enumerate(ty) if it in FBlist]
for i,xtick in enumerate(ax.get_yticklabels()):
    if i in tdx:
        xtick.set_color('r')

plt.plot([0,len(ty)],[0,len(ty)],color='w',linestyle='--')
#%% Recurrence matrix
pthresh = 25
cm = out_dict['con_mat']

cmthresh = (cm>pthresh).astype('int')
cmr = cmthresh.copy()
cmr2 = cmr.copy()
cmr2[cmr2==0] = 10000
cmr_plot = cmr-np.transpose(cmr2)
cmr_plot = (cmr_plot==0)*cm
z_dx = np.sum(cmr_plot,axis =0)>0
cmr_plot = cmr_plot[z_dx,:]
cmr_plot = cmr_plot[:,z_dx]
cluster_cmr,dmat_cmr = cf.hier_cosine(cmr_plot,0.7)
z_cmr = cf.linkage_order(cluster_cmr)
z_rank = np.argsort(-(np.sum(cmr_plot,axis=1)+np.sum(cmr_plot,axis=0)))

ty = out_dict['in_types'][z_dx][z_cmr]

cmr_plot = cmr_plot[z_cmr,:]
cmr_plot = cmr_plot[:,z_cmr]

plt.imshow(cmr_plot,vmax=200,interpolation='none')

plt.xticks(np.arange(0,len(ty)),labels=ty,rotation=90,fontsize=7)
ax = plt.subplot()

FBlist = ['FB6P','FB5J','FB6H','FB5H','FB4M','FB4L','FB4R','FB4X','FB4C','FB5I','FB5AB','FC2A','FC2C','FC2B','PFL3','hDeltaC','hDeltaB','hDeltaJ','FB4P_b']
L = ax.get_xticklabels()
tdx = [i for i,it in enumerate(ty) if it in FBlist]
for i,xtick in enumerate(L):
    if i in tdx:
        xtick.set_color('r')
plt.yticks(np.arange(0,len(ty)),labels=ty,fontsize=6)
plt.gcf().subplots_adjust(bottom=0.2) 

tdx = [i for i,it in enumerate(ty) if it in FBlist]
for i,xtick in enumerate(ax.get_yticklabels()):
    if i in tdx:
        xtick.set_color('r')

plt.plot([0,len(ty)],[0,len(ty)],color='w',linestyle='--')
#%% Feedforward matrix
plt.figure()
cm = out_dict['con_mat']
cmr = cmthresh.copy()
cmr2 = cmr.copy()
cmr2[cmr2==0] = 0
cmr_plot = cmr-np.transpose(cmr2)
cmr_plot = (cmr_plot!=0)*cm
#z_dx = np.sum(cmr_plot,axis =0)>0
cmr_plot = cmr_plot[z_dx,:]
cmr_plot = cmr_plot[:,z_dx]
#cluster_cmr,dmat_cmr = cf.hier_cosine(cmr_plot,0.7)
#z_cmr = cf.linkage_order(cluster_cmr)
z_rank = np.argsort(-np.sum(cmr_plot,axis=1))
ty = out_dict['in_types'][z_dx][z_cmr]

cmr_plot = cmr_plot[z_cmr,:]
cmr_plot = cmr_plot[:,z_cmr]

plt.imshow(cmr_plot,vmin=25,vmax=200,interpolation='none')

plt.xticks(np.arange(0,len(ty)),labels=ty,rotation=90,fontsize=8)
ax = plt.subplot()

FBlist = ['FB6P','FB5J','FB6H','FB5H','FB4M','FB4L','FB4R','FB4X','FB4C','FB5I','FB5AB','FC2A','FC2C','FC2B','PFL3','hDeltaC','hDeltaB','hDeltaJ','FB4P_b']
L = ax.get_xticklabels()
tdx = [i for i,it in enumerate(ty) if it in FBlist]
for i,xtick in enumerate(L):
    if i in tdx:
        xtick.set_color('r')
plt.yticks(np.arange(0,len(ty)),labels=ty,fontsize=6)
plt.gcf().subplots_adjust(bottom=0.2) 

tdx = [i for i,it in enumerate(ty) if it in FBlist]
for i,xtick in enumerate(ax.get_yticklabels()):
    if i in tdx:
        xtick.set_color('r')

plt.plot([0,len(ty)],[0,len(ty)],color='w',linestyle='--')

#%%
import networkx as nx
import itertools
g = nx.Graph(cm)
gv = nx.graphviews.generic_graph_view(g)
gv.edges(data=True)

comp =  nx.community.girvan_newman(g)
k=2
for communities in itertools.islice(comp, k):
    print(tuple(sorted(c) for c in communities))
    
    
    
    
#%%
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Load karate graph and find communities using Girvan-Newman
G = nx.Graph(cm)
communities = list(nx.community.girvan_newman(G))

# Modularity -> measures the strength of division of a network into modules
modularity_df = pd.DataFrame(
    [
        [k + 1, nx.community.modularity(G, communities[k])]
        for k in range(len(communities))
    ],
    columns=["k", "modularity"],
)


# function to create node colour list
def create_community_node_colors(graph, communities):
    number_of_colors = len(communities[0])
    colors = ["#D4FCB1", "#CDC5FC", "#FFC2C4", "#F2D140", "#BCC6C8"][:number_of_colors]
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors


# function to plot graph with node colouring based on communities
def visualize_communities(graph, communities, i):
    node_colors = create_community_node_colors(graph, communities)
    modularity = round(nx.community.modularity(graph, communities), 6)
    title = f"Community Visualization of {len(communities)} communities with modularity of {modularity}"
    pos = nx.spring_layout(graph, k=0.3, iterations=50, seed=2)
    plt.subplot(3, 1, i)
    plt.title(title)
    nx.draw(
        graph,
        pos=pos,
        node_size=1000,
        node_color=node_colors,
        with_labels=True,
        font_size=20,
        font_color="black",
    )


fig, ax = plt.subplots(3, figsize=(15, 20))

# Plot graph with colouring based on communities
visualize_communities(G, communities[0], 1)
visualize_communities(G, communities[3], 2)

# Plot change in modularity as the important edges are removed
modularity_df.plot.bar(
    x="k",
    ax=ax[2],
    color="#F2D140",
    title="Modularity Trend for Girvan-Newman Community Detection",
)
plt.show()