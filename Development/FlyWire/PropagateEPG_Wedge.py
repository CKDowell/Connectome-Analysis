# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:59:01 2024

@author: dowel


Aim of script is to propagate EPG wedge assignment forward to delta7, then downstream

Inputs:
    Neuron types
    
Outputs:
    Wedge orientation for each cell

"""

import caveclient
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from Utils.utils_general import utils_general as ug
import os
client = caveclient.CAVEclient()
#client.auth.setup_token(make_new=False)

my_token = "942d1c4bd3683cb7c09302a1d8430bb6"
client.auth.save_token(token=my_token,overwrite=True)

datastack_name = "flywire_fafb_public"
client = caveclient.CAVEclient(datastack_name)

client.materialize.get_tables()

NI = client.materialize.query_table("neuron_information_v2")

cell_class_type_annos_df = client.materialize.query_table("hierarchical_neuron_annotations", filter_in_dict={"classification_system": ["cell_class", "cell_type"]})
cell_class_type_annos_df
top_dir = 'D:\\ConnectomeData\\FlywireWholeBrain\\'
fn = top_dir + 'classification.csv'
df_meta = pd.read_csv(fn)

fn = top_dir + 'connections_princeton_no_threshold.csv'
df = pd.read_csv(fn)

fn = top_dir + 'neuropil_synapse_table.csv'
df_synreg = pd.read_csv(fn)
#%% Get EPG information
dx = df_meta['hemibrain_type']=='EPG'
print(np.sum(dx))
EPG = df_meta['root_id'][dx].to_numpy()

all_coords = np.array([])
presyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": EPG})
postsyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": EPG})
#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ebdx = presyn_df['neuropil']=='EB'
nids1 = presyn_df['pre_pt_root_id'][ebdx]
ebcoords1 = np.array(presyn_df['pre_pt_position'][ebdx].tolist())
#ax.scatter(ebcoords[:,0],ebcoords[:,1],ebcoords[:,2],s=1)
ebdx = postsyn_df['neuropil']=='EB'
nids = postsyn_df['post_pt_root_id'][ebdx]
ebcoords = np.array(postsyn_df['post_pt_position'][ebdx].tolist())
#ax.scatter(ebcoords[:,0],ebcoords[:,1],ebcoords[:,2],s=1)



mean_coords = np.zeros((len(EPG),3))
for i,n in enumerate(EPG):
    c = i+np.zeros(np.sum(nids==n))
   # plt.scatter(pbcoords[nids==n,0],pbcoords[nids==n,1],s=5)
    cmean = np.median(ebcoords[nids==n,:],axis=0)
    mean_coords[i,:] = cmean
    ax.scatter(cmean[0],cmean[1],cmean[2],color='k')
    
    cmean = np.median(ebcoords1[nids1==n,:],axis=0)
    ax.scatter(cmean[0],cmean[1],cmean[2],color='r')
#%% 
PBL = presyn_df['pre_pt_root_id'][presyn_df['neuropil']=='GA_R'].unique()
PBR = presyn_df['pre_pt_root_id'][presyn_df['neuropil']=='GA_L'].unique()
#%%
from sklearn.decomposition import PCA

# Get plane of EB
pca = PCA(n_components=3)
ebfit = (ebcoords-np.mean(ebcoords,axis=0))/np.std(ebcoords,axis=0)
pca.fit(ebfit)
b = pca.components_
proj = np.matmul(ebfit,pca.components_.T)


pbdx = presyn_df['neuropil']=='PB'
pids = presyn_df['pre_pt_root_id'][pbdx]
pbcoords = np.array(presyn_df['pre_pt_position'][pbdx].tolist())

mean_coords = np.zeros((len(EPG),3))
mean_coords_pb = np.zeros((len(EPG),3))
for i,n in enumerate(EPG):
    cmean = np.median(proj[nids==n,:],axis=0)
    mean_coords[i,:] = cmean
    pmean = np.median(pbcoords[pids==n,:],axis=0)
    mean_coords_pb[i,:] = pmean
anat_phase = np.arctan2(mean_coords[:,0],mean_coords[:,1])

ap_deg = 180*(anat_phase+np.pi)/np.pi
ap_deg = np.round(ap_deg)

plt.subplot(1,2,1)
c = np.zeros(len(proj))
cp = np.zeros(len(pbcoords))
for i,n in enumerate(EPG):
    c[nids==n] = anat_phase[i]
    cp[pids==n] = anat_phase[i]

plt.scatter(proj[:,0],proj[:,1],s=1,c=c,alpha=0.1,cmap='coolwarm')
plt.scatter(mean_coords[:,0],mean_coords[:,1],c=anat_phase,cmap='coolwarm')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pbcoords[:,0],pbcoords[:,1],pbcoords[:,2],c=cp,s=1,alpha=0.1,cmap='coolwarm')
ax.scatter(mean_coords_pb[:,0],mean_coords_pb[:,1],mean_coords_pb[:,2],c=anat_phase,cmap='coolwarm')
for i,ip in enumerate(ap_deg):
    ax.text(mean_coords_pb[i,0],mean_coords_pb[i,1],mean_coords_pb[i,2],ip)
    


tL = EPG[np.in1d(EPG,PBL)]
tLtheta = anat_phase[np.in1d(EPG,PBL)]
dx = np.argsort(tLtheta)
tL = tL[dx]
tLtheta = tLtheta[dx]
tR = EPG[np.in1d(EPG,PBR)]
tRtheta = anat_phase[np.in1d(EPG,PBR)]
dx = np.argsort(tRtheta)
tR = tR[dx]
tRtheta  =tRtheta[dx]
ranked_EPG = np.append(tL[:],tR[:])
ranked_theta = np.append(tLtheta,tRtheta)

ranked_pbcoords = np.zeros((len(ranked_EPG),3))
for i,n in enumerate(ranked_EPG):
    pmean = np.median(pbcoords[pids==n,:],axis=0)
    ranked_pbcoords[i,:] = pmean


pre_EPG = postsyn_df['pre_pt_root_id'].unique()

#%% Glomerular identification from clustering
plt.close('all')
cludata = np.append(ranked_pbcoords,np.cos(ranked_theta[:,np.newaxis]),axis=1)
cludata = np.append(cludata,np.sin(ranked_theta[:,np.newaxis]),axis=1)

cludata = (cludata-np.mean(cludata,axis=0))/np.std(cludata,axis=0)
cludata[:,3:] = cludata[:,3:]*2 # Increase the weight of the EB phase to ensure reliable glomerular assignment
from sklearn.cluster import KMeans
from scipy.stats import circmean
km = KMeans(n_clusters=16,n_init=50).fit(cludata) # Does not always give appropriate coords
cents = km.cluster_centers_
cdx = np.argsort(cents[:,0])
cents = cents[cdx,:]


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(cludata[:,0],cludata[:,1],cludata[:,2],color='k')
ax.scatter(cents[:,0],cents[:,1],cents[:,2],color='r')

cs = np.argsort(cents[:,0])
cents  = cents[cs,:]
labs = np.array([])
for i,c in enumerate(cludata):
    l = np.argmin(np.sum((cents-c)**2,axis=1))
    labs = np.append(labs,l)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

dx_PB_glom = np.array([],dtype='int')
glom_id = np.array([],dtype='int')
glom_theta = np.array([])
glomorder = np.arange(0,16)
glomorder[0] = 1 
glomorder[1] = 0
for io,i in enumerate(glomorder):
    dx_PB_glom = np.append(dx_PB_glom,np.where(labs==i)[0])
    glom_id = np.append(glom_id,np.ones(np.sum(labs==i),dtype='int')*io)
    ax.scatter(cludata[labs==i,0],cludata[labs==i,1],cludata[labs==i,2],color='r')
    ax.text(cents[i,0],cents[i,1],cents[i,2],io)
    glom_theta = np.append(glom_theta,circmean(ranked_theta[labs==i],low=-np.pi,high=np.pi))

ranked_EPG_final = ranked_EPG[dx_PB_glom]
ranked_theta_final = ranked_theta[dx_PB_glom]
ranked_glom_theta = glom_theta[glom_id]
ranked_pbcoords_final = ranked_pbcoords[dx_PB_glom,:]

# Save this information - consolidate the above to minimal set
savedict = {'root_ids':ranked_EPG_final,'EB_theta':ranked_theta_final,
            'PB_glomeruli':glom_id,'glom_theta':ranked_glom_theta,'Med_PB_coords':ranked_pbcoords_final}
savedir = 'D:\\ConnectomeData\\FlywireWholeBrain'
ug.save_pick(savedict,os.path.join(savedir,'EPG_GlomAdv.pkl'))
# Then work on
#%% Propagate thru to delta7

dx = df_meta['hemibrain_type']=='Delta7'
print(np.sum(dx))
delta7 = df_meta['root_id'][dx].to_numpy()
presyn_d7 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": delta7})
postsyn_d7 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": delta7})
#%% 
EPG_dict = ug.load_pick(os.path.join(savedir,'EPG_GlomAdv.pkl'))
# Group based upon two criteria

# 1. Output glomerulous - there are 18 rather than 16 glomeruli 

pbdx = presyn_d7['neuropil']=='PB'
pids = presyn_d7['pre_pt_root_id'][pbdx]
d7_pbcoords = np.array(presyn_d7['pre_pt_position'][pbdx].tolist())

d7_pb_out = np.zeros((len(delta7),3,3))

for i,c in enumerate(delta7):
    
    km = KMeans(n_clusters=3).fit(d7_pbcoords[pids==c,:]) # Kmeans cluster with three because some span glom -1 and glom 16
    
    cents = km.cluster_centers_
    co = np.argsort(cents[:,0])
    d7_pb_out[i,:,0] = cents[co[0],:]
    d7_pb_out[i,:,1] = cents[co[1],:]
    d7_pb_out[i,:,2] = cents[co[2],:]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(d7_pb_out[:,0,0],d7_pb_out[:,1,0],d7_pb_out[:,2,0],color='b')
ax.scatter(d7_pb_out[:,0,1],d7_pb_out[:,1,1],d7_pb_out[:,2,1],color='r')
ax.scatter(d7_pb_out[:,0,2],d7_pb_out[:,1,2],d7_pb_out[:,2,2],color='r')


# 2. Input EPG tuning
input_thetas = np.zeros(len(delta7))
input_thetas_norm = np.zeros(len(delta7))
normvals = np.linspace(-np.pi,np.pi,16)
for i,c in enumerate(delta7):
    tdf = postsyn_d7[postsyn_d7['post_pt_root_id']==c]
    dx = np.in1d(tdf['pre_pt_root_id'],EPG_dict['root_ids'])
    tepg = tdf['pre_pt_root_id'][dx].unique()
    weights = np.zeros(len(tepg))
    thetas = np.zeros(len(tepg))
    gloms = np.zeros(len(tepg),dtype='int')
    for it,ep in enumerate(tepg):
        weights[it] = np.sum(tdf['pre_pt_root_id']==ep)
        edx = EPG_dict['root_ids']==ep
        thetas[it] = EPG_dict['EB_theta'][edx][0]
        gloms[it] = EPG_dict['PB_glomeruli'][edx][0]
    weights = weights/np.sum(weights)
    thetas_norm = normvals[gloms]
    
    tsin = weights*np.sin(thetas)
    tcos = weights*np.cos(thetas)
    tsinsum =np.sum(tsin)
    tcossum = np.sum(tcos)
    input_thetas[i] = np.arctan2(tsinsum,tcossum)
    
    tsin = weights*np.sin(thetas_norm)
    tcos = weights*np.cos(thetas_norm)
    tsinsum =np.sum(tsin)
    tcossum = np.sum(tcos)
    input_thetas_norm[i] = np.arctan2(tsinsum,tcossum)

c1 = np.zeros(len(d7_pbcoords))
c2 = np.zeros(len(d7_pbcoords))
for i,c in enumerate(delta7):
    c1[pids==c] = input_thetas[i]
    c2[pids==c] = input_thetas_norm[i]
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(d7_pbcoords[:,0],d7_pbcoords[:,1],d7_pbcoords[:,2],c=c1,cmap='coolwarm',s=1)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(d7_pbcoords[:,0],d7_pbcoords[:,1],d7_pbcoords[:,2],c=c2,cmap='coolwarm',s=1)

#%%
savedict = {'root_ids':delta7,'PB_theta':input_thetas,'PB_theta_by_glom':input_thetas_norm,
            'output_locos':d7_pb_out}
savedir = 'D:\\ConnectomeData\\FlywireWholeBrain'
ug.save_pick(savedict,os.path.join(savedir,'Delta7_GlomAdv.pkl'))

#%%



for i in range(3):
    clumini = np.append(d7_pb_out[:,:,i],np.cos(input_thetas_norm[:,np.newaxis]),axis=1)
    clumini = np.append(clumini,np.sin(input_thetas_norm[:,np.newaxis]),axis=1)
    if i==0:
        cludata7 = clumini
    else:
        cludata7 = np.append(cludata7,clumini,axis=0)








cludata7 =( cludata7-np.mean(cludata7,axis=0))/np.std(cludata7,axis=0)
cludata7[:,3:] = cludata7[:,3:]*2
km = KMeans(n_clusters=18).fit(cludata7)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(cludata7[:,0],cludata7[:,1],cludata7[:,2],c=np.append(input_thetas_norm,np.append(input_thetas_norm,input_thetas_norm)),cmap='coolwarm')
ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2],color='k')
#  2.a Via the standard glomerulous notation
#  2.b Via the theta values of the EPG neurons themselves

#https://scikit-learn.org/1.5/modules/manifold.html