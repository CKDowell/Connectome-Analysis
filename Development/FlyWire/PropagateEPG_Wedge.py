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
from Utilities.utils_general import utils_general as ug
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

fn = top_dir + 'fafb_v783_princeton_synapse_table.csv.gz'
df_p = pd.read_csv(fn) 

# Note on data. For the FAFB the anatomical labels are correctly left/right assigned. The coordinates in
# 3D renders online are LR flipped. Higher x coordinates indicate the right hand side, lower left hand side
#%% Get EPG information
dx = df_meta['hemibrain_type']=='EPG'
print(np.sum(dx))
EPG = df_meta['root_id'][dx].to_numpy()
EPG = np.mod(EPG,720575940000000000)
all_coords = np.array([])
# presyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": EPG})
# postsyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": EPG})
presyn_df = df_p[np.in1d(df_p['pre_root_id_720575940'],EPG)]
postsyn_df = df_p[np.in1d(df_p['post_root_id_720575940'],EPG)]
#%%
prelist = ['pre_x','pre_y','pre_z']
postlist = ['post_x','post_y','post_z']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ebdx = presyn_df['neuropil']=='EB'
nids1 = presyn_df['pre_root_id_720575940'][ebdx] # edit to bring CAVE back
ebcoords1 = np.array(presyn_df[prelist][ebdx].to_numpy())
ebcoords1[:,0] = ebcoords1[:,0] # flip x axis so left is lower than right

#ax.scatter(ebcoords[:,0],ebcoords[:,1],ebcoords[:,2],s=1)
ebdx = postsyn_df['neuropil']=='EB'
nids = postsyn_df['post_root_id_720575940'][ebdx]
ebcoords = np.array(postsyn_df[postlist][ebdx].to_numpy())
ebcoords[:,0] = ebcoords[:,0]
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
#%% Get PB side from Gall projections
PBL = presyn_df['pre_root_id_720575940'][presyn_df['neuropil']=='GA_R'].unique()
PBR = presyn_df['pre_root_id_720575940'][presyn_df['neuropil']=='GA_L'].unique()
#%%
from sklearn.decomposition import PCA

# Get plane of EB
pca = PCA(n_components=3)
ebfit = (ebcoords-np.mean(ebcoords,axis=0))/np.std(ebcoords,axis=0)
pca.fit(ebfit)
b = pca.components_
proj = np.matmul(ebfit,pca.components_.T)


pbdx = presyn_df['neuropil']=='PB'
pids = presyn_df['pre_root_id_720575940'][pbdx]
#pbcoords = np.array(presyn_df['pre_pt_position'][pbdx].tolist())
pbcoords = np.array(presyn_df[prelist][pbdx].to_numpy())
pbcoords[:,0] = pbcoords[:,0]

mean_coords = np.zeros((len(EPG),3))
mean_coords_pb = np.zeros((len(EPG),3))
mean_coords_raw = np.zeros((len(EPG),3))
#Calculate mean EB and PB coordinates for EB phase and glomerulous determination
for i,n in enumerate(EPG):
    cmean = np.mean(proj[nids==n,:],axis=0)
    mean_coords[i,:] = cmean
    mean_coords_raw[i,:] = np.mean(ebcoords[nids==n,:],axis=0)
    
    pmean = np.mean(pbcoords[pids==n,:],axis=0)
    mean_coords_pb[i,:] = pmean
    

# EB phase determination from PCA projection.
# 2 adjustments
# 1. negative x values, so that right side has negative
# 2. subtract pi/2 so that bottom of EB has +-180 degrees as per imaging convention
anat_phase = ug.circ_subtract(np.arctan2(mean_coords[:,0],mean_coords[:,1]),np.pi/2) # have to add additional 90 degrees since PCA rotates the EB



ap_deg = 180*(anat_phase)/np.pi
ap_deg = np.round(ap_deg)
plt.figure()
plt.subplot(1,2,1)
plt.title('PCA space')
c = np.zeros(len(proj))
cp = np.zeros(len(pbcoords))
for i,n in enumerate(EPG):
    c[nids==n] = anat_phase[i]
    cp[pids==n] = anat_phase[i]

plt.scatter(proj[:,0],proj[:,1],s=1,c=c,alpha=0.1,cmap='coolwarm')
plt.subplot(1,2,2)
plt.title('Anat space')
plt.scatter(mean_coords_raw[:,0],mean_coords_raw[:,2],c=anat_phase,cmap='coolwarm')
for i,ip in enumerate(ap_deg):
    plt.text(mean_coords_raw[i,0],mean_coords_raw[i,2],str(np.round(ip)))

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


pre_EPG = postsyn_df['pre_root_id_720575940'].unique()

#%% Glomerular identification from clustering
# Glomeruli are assigned by doing kmeans clustering on the mean presyn PB positions
# and also the neuronal phase assignment as determined from the ellipsoid body

plt.close('all')
cludata = np.append(ranked_pbcoords,np.cos(ranked_theta[:,np.newaxis]),axis=1)
cludata = np.append(cludata,np.sin(ranked_theta[:,np.newaxis]),axis=1)

cludata = (cludata-np.mean(cludata,axis=0))/np.std(cludata,axis=0)
cludata[:,3:] = cludata[:,3:]*3 # Increase the weight of the EB phase to ensure reliable glomerular assignment
from sklearn.cluster import KMeans
from scipy.stats import circmean
km = KMeans(n_clusters=16,n_init=50).fit(cludata) # Does not always give appropriate coords

lab_o = km.labels_
counts = np.bincount(lab_o)
single_cle = np.where(counts==0)[0]


cents = km.cluster_centers_
cdx = np.argsort(-cents[:,0])
cents = cents[cdx,:]


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(cludata[:,0],cludata[:,1],cludata[:,2],color='k')
ax.scatter(cents[:,0],cents[:,1],cents[:,2],color='r')

cs = np.argsort(-cents[:,0])
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




# Glomeruli are counted from right to left as per my convention

# Minor adjustment of array order
glomorder[14] = 15
glomorder[15] = 14
for io,i in enumerate(glomorder):
    dx_PB_glom = np.append(dx_PB_glom,np.where(labs==i)[0])
    glom_id = np.append(glom_id,np.ones(np.sum(labs==i),dtype='int')*io)
    ax.scatter(ranked_pbcoords[labs==i,0],ranked_pbcoords[labs==i,1],ranked_pbcoords[labs==i,2],color='r')
    #ax.text(cents[i,0],cents[i,1],cents[i,2],io)
    ax.text(ranked_pbcoords[labs==i,0][0],ranked_pbcoords[labs==i,1][0],ranked_pbcoords[labs==i,2][0],io)
    glom_theta = np.append(glom_theta,circmean(ranked_theta[labs==i],low=-np.pi,high=np.pi))

ranked_EPG_final = ranked_EPG[dx_PB_glom]
ranked_theta_final = ranked_theta[dx_PB_glom]
ranked_glom_theta = glom_theta[glom_id]
ranked_pbcoords_final = ranked_pbcoords[dx_PB_glom,:]

plt.figure()
plt.plot(ranked_pbcoords_final[:,0],glom_id)

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
delta7 = np.mod(delta7,720575940000000000)



# presyn_d7 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": delta7})
# postsyn_d7 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": delta7})

presyn_d7 = df_p[np.in1d(df_p['pre_root_id_720575940'],delta7)]
postsyn_d7 = df_p[np.in1d(df_p['post_root_id_720575940'],delta7)]
#%% Determine anatomical features and inherited tuning of delta 7 neurons
EPG_dict = ug.load_pick(os.path.join(savedir,'EPG_GlomAdv.pkl'))
# Group based upon two criteria

# 1. Output glomerulous - there are 18 rather than 16 glomeruli 

pbdx = presyn_d7['neuropil']=='PB'
pids = presyn_d7['pre_root_id_720575940'][pbdx]
#d7_pbcoords = np.array(presyn_d7['pre_pt_position'][pbdx].tolist())
d7_pbcoords = np.array(presyn_d7[prelist][pbdx].to_numpy())

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
input_gloms = np.zeros((16,len(delta7)))
normvals = np.linspace(-np.pi,np.pi,16)
for i,c in enumerate(delta7):
    # Find a cell
    tdf = postsyn_d7[postsyn_d7['post_root_id_720575940']==c]
    # Get its EPG inputs
    dx = np.in1d(tdf['pre_root_id_720575940'],EPG_dict['root_ids'])
    tepg = tdf['pre_root_id_720575940'][dx].unique()
    
    weights = np.zeros(len(tepg))
    thetas = np.zeros(len(tepg))
    gloms = np.zeros(len(tepg),dtype='int')
    # Collect synapse counts and anatomical theta of EPG inputs
    for it,ep in enumerate(tepg):
        weights[it] = np.sum(tdf['pre_root_id_720575940']==ep)
        edx = EPG_dict['root_ids']==ep
        thetas[it] = EPG_dict['EB_theta'][edx][0]
        gloms[it] = EPG_dict['PB_glomeruli'][edx][0]
        input_gloms[gloms[it],i] = input_gloms[gloms[it],i]+weights[it]
    weights = weights/np.sum(weights)
    thetas_norm = normvals[gloms]
    
    tsin = weights*np.sin(thetas)
    tcos = weights*np.cos(thetas)
    tsinsum =np.sum(tsin)
    tcossum = np.sum(tcos)
    input_thetas[i] = np.arctan2(tsinsum,tcossum)
    
    tsin = weights*np.sin(thetas_norm) # Normalised/idealised values applied to gloms
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
#Values derived straight from EB (input thetas)
ax = fig.add_subplot(projection='3d')
ax.scatter(d7_pbcoords[:,0],d7_pbcoords[:,1],d7_pbcoords[:,2],c=c1,cmap='coolwarm',s=1)

fig = plt.figure()
#Values idealised (input thetas norm)
ax = fig.add_subplot(projection='3d')
ax.scatter(d7_pbcoords[:,0],d7_pbcoords[:,1],d7_pbcoords[:,2],c=c2,cmap='coolwarm',s=1)

#%%
savedict = {'root_ids':delta7,'PB_theta':input_thetas,'PB_theta_by_glom':input_thetas_norm,
            'output_locos':d7_pb_out,'input_gloms':input_gloms}
savedir = 'D:\\ConnectomeData\\FlywireWholeBrain'
ug.save_pick(savedict,os.path.join(savedir,'Delta7_GlomAdv.pkl'))

#%% PFL3 neurons
dx = df_meta['hemibrain_type']=='PFL3'
print(np.sum(dx))
PFL3 = df_meta['root_id'][dx].to_numpy()

PFL3 = np.mod(PFL3,720575940000000000)



# presyn_d7 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": delta7})
# postsyn_d7 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": delta7})

presyn_PFL3= df_p[np.in1d(df_p['pre_root_id_720575940'],PFL3)]
postsyn_PFL3 = df_p[np.in1d(df_p['post_root_id_720575940'],PFL3)]

#presyn_PFL3 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": PFL3})
#postsyn_PFL3 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": PFL3})

#%% Get PFL FSB columnar arrangement
postlist = ['post_x','post_y','post_z']
from scipy import stats
fsbdx = postsyn_PFL3['neuropil']=='FB'
pfl3coords = np.zeros((len(PFL3),3))
pfl3_projmean = np.zeros((len(PFL3),3))
pfl3_coords_all = postsyn_PFL3[postlist].to_numpy()
pfl3_coords_all = np.array(pfl3_coords_all.tolist())
pfl3_coords_all[:,0] = pfl3_coords_all[:,0]


pca = PCA(n_components=3)
pfl3fit = (pfl3_coords_all-np.mean(pfl3_coords_all,axis=0))/np.std(pfl3_coords_all,axis=0)
pca.fit(pfl3fit)
b = pca.components_
proj = np.matmul(pfl3fit,pca.components_.T)
    
for i,p in enumerate(PFL3):
    pdx = postsyn_PFL3['post_root_id_720575940']==p
    dx = np.logical_and(pdx,fsbdx)
    tcoords = postsyn_PFL3[postlist][dx].to_numpy()
    tcoords = np.array(tcoords.tolist())
    tcoords[:,0] = tcoords[:,0]
    pfl3coords[i,:] = np.mean(tcoords,axis=0)
    pfl3_projmean[i,:] = np.mean(proj[dx,:],axis=0)
    
km = KMeans(n_clusters=12,n_init=50).fit(pfl3coords) # Does not always give appropriate coords
cents = km.cluster_centers_
cr = np.argsort(-cents[:,0]) # rank so that right side are ranked first (ie descending order)
cents_ranked = cents[cr,:]
col_array= np.arange(0,12)
col_array2 = col_array.copy()
col_array2[cr] = col_array.copy()
pfl3_col_id = col_array2[km.labels_] 

# Get FSB column thetas from PB thetas, numbered right to left
pb_rank_cond = np.append(np.arange(0,8),np.arange(0,8))
pb_rank_cond_9 = np.array([1,2,3,4,5,6,7,0,8,1,2,3,4,5,6,7])

col9_theta = np.zeros(9)
col8_theta = np.zeros(8)
for i in range(8):
    col8_theta[i] = stats.circmean(glom_theta[pb_rank_cond==i],low=-np.pi,high=np.pi)
for i in range(9):
    col9_theta[i] = stats.circmean(glom_theta[pb_rank_cond_9==i],low=-np.pi,high=np.pi)
# This shift if zeroed produces a perfect PFL3 output... needto check...
col8_theta_shift = np.roll(col8_theta,0) # shift by one column since middle glom innervates right FSB


col12_theta = ug.circ_interp(np.linspace(0,7,12),np.arange(0,8),col8_theta_shift)
col12_theta2 = ug.circ_interp(np.linspace(0,8,12),np.arange(0,9),col9_theta)
pfl3_thetas = col12_theta2[pfl3_col_id]
    
    
    
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pfl3coords[:,0],pfl3coords[:,1],pfl3coords[:,2])

ax.scatter(cents[:,0],cents[:,1],cents[:,2],color='r')

for ir,r in enumerate(pfl3_col_id):
    theta = np.round((180/np.pi)*pfl3_thetas[ir])
    ax.text(pfl3coords[ir,0],pfl3coords[ir,1],pfl3coords[ir,2],str(theta))

ax.scatter(ranked_pbcoords_final[:,0],ranked_pbcoords_final[:,1],ranked_pbcoords_final[:,2],color='k')
for ir,r in enumerate(ranked_pbcoords_final):
    theta =np.round((180/np.pi)*ranked_theta_final[ir])
    ax.text(r[0],r[1],r[2],theta)
    #%%
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# #%%
# tcoords = pfl3_coords_all[np.in1d(postsyn_PFL3['post_pt_root_id'],PFL3[PFL3_LAL==-1]),:]
    
# ax.scatter(tcoords[:,0],tcoords[:,1],tcoords[:,2])
#%% Get PFL3 EPG, delta7 input matrix
PFL3_inputmat = np.zeros((len(ranked_EPG_final)+len(delta7),len(PFL3)))
PFL3_LAL = np.zeros(len(PFL3))

PFL3_input_angles = np.append(ranked_theta_final,input_thetas)
activation_sign = np.append(np.ones(len(ranked_theta_final)),-np.ones(len(input_thetas)))

#PFL3_input_gloms = np.append(glom_id,)

for i,p in enumerate(PFL3):
    pdx = postsyn_PFL3['post_root_id_720575940']==p
    ROIs = np.unique(postsyn_PFL3['neuropil'][pdx].dropna().to_numpy())
    if sum(ROIs=='LAL_L')==1:
        PFL3_LAL[i] = -1 
    elif sum(ROIs=='LAL_R')==1:
        PFL3_LAL[i] = 1
    for ie,e in enumerate(ranked_EPG_final):
        edx = postsyn_PFL3['pre_root_id_720575940']==e
        dx = np.logical_and(pdx,edx)
        PFL3_inputmat[ie,i] = np.sum(dx)
        
    for ie,e in enumerate(delta7):
        edx = postsyn_PFL3['pre_root_id_720575940']==e
        dx = np.logical_and(pdx,edx)
        PFL3_inputmat[ie+len(ranked_EPG_final),i] = np.sum(dx)

#%% Save data
savedict = {'col12_theta': col12_theta2,'PFL3_thetas':pfl3_thetas,'root_ids': PFL3, 
            'PFL3_input_angles': PFL3_input_angles,'PFL3_inputmat':PFL3_inputmat,'activation_sign': activation_sign,
            'PFL3_LAL':PFL3_LAL,'PFL3_col_id': pfl3_col_id}
savedir = 'D:\\ConnectomeData\\FlywireWholeBrain'
ug.save_pick(savedict,os.path.join(savedir,'PFL3_info.pkl'))
#%% Model output for different EB offsets without FSB input
#plt.close('all')
offsets = np.linspace(-np.pi,np.pi,100)
diff = np.zeros(len(offsets))
PFL3_inputmatN = PFL3_inputmat/np.sum(PFL3_inputmat,axis=0)
for io,o in enumerate(offsets):
    act_vector = (np.cos(PFL3_input_angles-o)+1)*activation_sign
    pact = np.matmul(act_vector,PFL3_inputmat)
    L = np.sum(pact[PFL3_LAL==-1])
    R = np.sum(pact[PFL3_LAL==1])
    plt.figure(101)
    plt.scatter(o,L,color='b')
    plt.scatter(o,R,color='r')
    diff[io] = R-L
    
    
plt.figure(101)
plt.xlabel('Bump position (rads)')
plt.ylabel('Left/Right PFL activation (AU)')
#plt.ylim([-1200,-500])
plt.figure(102)
plt.scatter(offsets,diff,c=offsets,cmap='coolwarm')    
#plt.ylim([-600,600])
plt.ylabel('Right - Left PFL activation (AU)')
plt.xlabel('EPG Bump position (rads)')

#%% Model output for different EB offsets without FSB input
# When EPG bump is to the left it should have a right turn
#plt.close('all')
offsets = np.linspace(-np.pi,np.pi,100)
diff = np.zeros(len(offsets))
PFL3_inputmatN = PFL3_inputmat.copy()
d7_epg_rat =np.sum(PFL3_inputmat[:len(ranked_EPG_final),:]) /np.sum(PFL3_inputmat[:])
PFL3_inputmatN[:len(ranked_EPG_final),:] = d7_epg_rat*PFL3_inputmatN[:len(ranked_EPG_final),:]/np.sum(PFL3_inputmatN[:len(ranked_EPG_final),:],axis=0) 
PFL3_inputmatN[len(ranked_EPG_final):,:] = (1-d7_epg_rat)*PFL3_inputmatN[len(ranked_EPG_final):,:]/np.sum(PFL3_inputmatN[len(ranked_EPG_final):,:],axis=0) 

for io,o in enumerate(offsets):
    act_vector = (np.cos(PFL3_input_angles-o)+1)*activation_sign
    pact = np.matmul(act_vector,PFL3_inputmatN)
    L = np.sum(pact[PFL3_LAL==-1])
    R = np.sum(pact[PFL3_LAL==1])
    plt.figure(201)
    plt.scatter(o,L,color='b')
    plt.scatter(o,R,color='r')
    diff[io] = (R-L)
    
    
plt.figure(201)
plt.xlabel('Bump position (rads)')
plt.ylabel('Left/Right PFL activation (AU)')
#plt.ylim([-1200,-500])
plt.figure(202)
plt.scatter(offsets,diff,c=offsets,cmap='coolwarm')    
#plt.ylim([-600,600])
plt.ylabel('Right - Left PFL activation (AU)')
plt.xlabel('Bump position (rads)')
#%% Model output with FSB inputs as cosine with differing activities
plt.close('all')
offsetsH = np.linspace(-np.pi,np.pi,101)
offsetsG = np.linspace(-np.pi,np.pi,101)
diff = np.zeros((len(offsetsH),len(offsetsG)))
PFL3_inputmatN = PFL3_inputmat/np.sum(PFL3_inputmat,axis=0)
gstrengths = [0,.01,0.02,0.05,.1,1,2,4,8]
for ig,gstrength in enumerate(gstrengths):
    for io,o in enumerate(offsetsH):
        act_vector = (np.cos(PFL3_input_angles-o)+1)*activation_sign
        pact = np.matmul(act_vector,PFL3_inputmatN)
        
        for i2,o2 in enumerate(offsetsG):
            gact =gstrength*(np.cos(pfl3_thetas-o2)+1)
            tact = np.exp(pact+gact)
            L = np.sum(tact[PFL3_LAL==-1])
            R = np.sum(tact[PFL3_LAL==1])
            diff[io,i2] = (R-L)/np.sum(R+L)
            #print(L,R)
    
    plt.subplot(3,3,ig+1)
    plt.title(gstrength)
    plt.imshow(np.flipud(diff),aspect='auto',cmap='coolwarm',vmin=-.1,vmax=.1)
    plt.xlabel('FSB bump')
    plt.ylabel('PB bump')
    labs = np.round((180/np.pi)*offsetsG[np.linspace(0,100,5,dtype='int')])
    plt.xticks(np.linspace(0,100,5),labels = labs)
    plt.yticks(np.linspace(0,100,5),labels = np.flipud(labs))
    plt.plot([0,100],[100,0],color='k')
#%% Model output with FSB inputs as bump that gets broader
plt.close('all')
offsetsH = np.linspace(-np.pi,np.pi,101)
offsetsG = np.linspace(-np.pi,np.pi,101)
diff = np.zeros((len(offsetsH),len(offsetsG)))
PFL3_inputmatN = PFL3_inputmat/np.sum(PFL3_inputmat,axis=0)
gstrengths = [0,0.02,0.05,.1,.2,.5,1,2,4]
for ig,gstrength in enumerate(gstrengths):
    for io,o in enumerate(offsetsH):
        act_vector = (np.cos(PFL3_input_angles-o)+1)*activation_sign
        pact = np.matmul(act_vector,PFL3_inputmatN)
        
        for i2,o2 in enumerate(offsetsG):
            gact =(np.cos(pfl3_thetas-o2)*gstrength+1)+5
            tact = np.exp(pact+gact)
            L = np.sum(tact[PFL3_LAL==-1])
            R = np.sum(tact[PFL3_LAL==1])
            diff[io,i2] = (R-L)/np.sum(R+L)
            #print(L,R)
    plt.figure(101)
    plt.subplot(3,3,ig+1)
    plt.imshow(np.flipud(diff),aspect='auto',cmap='coolwarm',vmin=-.5,vmax=.5)
    plt.xlabel('FSB bump')
    plt.ylabel('PB bump')
    labs = np.round((180/np.pi)*offsetsG[np.linspace(0,100,5,dtype='int')])
    plt.xticks(np.linspace(0,100,5),labels = labs)
    plt.yticks(np.linspace(0,100,5),labels = np.flipud(labs))
    plt.plot([0,100],[100,0],color='k')
    # diffroll = np.zeros_like(diff)
    # for i in range(101):
    #     diffroll[:,i] = np.roll(diff[:,i],-i)
        
    # plt.subplot(1,2,2)
    # plt.imshow(np.flipud(diffroll),aspect='auto',cmap='coolwarm')
    # plt.xlabel('FSB bump')
    # plt.ylabel('PB bump')
    # labs = np.round((180/np.pi)*offsetsG[np.linspace(0,100,5,dtype='int')])
    # plt.xticks(np.linspace(0,100,5),labels = labs)
    # plt.yticks(np.linspace(0,100,5),labels = np.flipud(labs))
    # #plt.plot([0,100],[100,0],color='k')
    # plt.colorbar()
plt.figure()
o2=0
for ig,gstrength in enumerate(gstrengths):
    gact =(np.cos(pfl3_thetas-o2)*gstrength+1)+5
    plt.scatter(pfl3_thetas,gact,label=ig)
plt.ylim([0,15])
plt.legend()
#%%

from analysis_funs.CX_analysis_col import CX_a
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper
from CD_edge_tracking.Models.neuro2pfl3 import pfl3_model
import matplotlib.pyplot as plt
datadir ="Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"
etp = ET_paper(datadir)
#%%
mdl = pfl3_model()
phase_eb = etp.cxa.pdat['phase_eb']
phase_goal = etp.cxa.pdat['phase_fsb_upper']


L,R,turns = mdl.model_pfl3_phase(phase_eb,phase_goal,goal_weight=2,eb_function='cosine',goal_function='cosine')
    
plt.plot(turns,color='k')    
fsb  = etp.cxa.pdat['wedges_fsb_upper']

L,R,turns2 =  mdl.model_pfl3_fsb_data(phase_eb,fsb,goal_weight=5,eb_function='cosine',goal_function='cosine')
plt.plot(turns2,color='r')
    
ebw = etp.cxa.pdat['wedges_eb']
L,R,turns3 = mdl.model_pfl3_all_data(ebw,fsb,goal_weight=10,d7weight=3)

plt.plot(turns3,color='b')

plt.figure()
plt.scatter(turns2,turns3,s=1)
etp.cxa.plot_traj_arrow_heat(['fsb_upper'],turns2,cmin=-.5,cmax=.5,a_sep=5)
#%% Play in some activity to the network
from analysis_funs.CX_analysis_col import CX_a
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper

datadir ="Y:\Data\FCI\Hedwig\FC2_maimon2\\240514\\f1\\Trial2"
etp = ET_paper(datadir)
#%% Model with EB phase, fsb phase or wedges
plt.close('all')
from scipy.interpolate import interpn
poffset = ug.circ_subtract(col12_theta2[0],-np.pi)
eb = ug.circ_subtract(etp.cxa.pdat['phase_eb'],-poffset) # add phase offset to match anatomy, this should be close to zero. From my data it is 18 degrees
phase = ug.circ_subtract(etp.cxa.pdat['phase_fsb_upper'],-poffset)
fsb = etp.cxa.pdat['wedges_fsb_upper']
#fsb = np.linspace(-np.pi,np.pi,16)[np.newaxis,:]
#fsb = np.tile(fsb,(len(eb),1))
#fsb = np.fliplr(np.cos(fsb-etp.cxa.pdat['phase_fsb_upper'][:,np.newaxis]))

x_old = np.linspace(0, 1, 16)
x_new = np.linspace(0, 1, 12)

fsb_interp =ug.circular_column_interp(fsb,x_old,x_new)
# np.apply_along_axis(lambda row: np.interp(x_new, x_old, row), 1, fsb)
turn = np.zeros(len(eb))
turn_rough = np.zeros(len(eb))
for ie,e in enumerate(eb):
    act_vector = (np.cos(PFL3_input_angles-e)+1)*activation_sign
    pact = np.matmul(act_vector,PFL3_inputmatN)
    
    gact = 2*(np.cos(-phase[ie]+pfl3_thetas)+1)
    tact = np.exp(pact+gact)
    
    
    L = np.sum(tact[PFL3_LAL==-1])
    R = np.sum(tact[PFL3_LAL==1])
    turn[ie]= (R-L)/np.sum(R+L)
    
    gact2 = fsb_interp[ie,pfl3_col_id]*5
    gact3 = 2*(fsb_interp[ie,pfl3_col_id]+1)
    
    tact = np.exp(pact+gact2)
    
    L = np.sum(tact[PFL3_LAL==-1])
    R = np.sum(tact[PFL3_LAL==1])
    turn_rough[ie]= (R-L)/np.sum(R+L)
    if np.mod(ie,1000)==0:
        plt.figure()
        plt.scatter(pfl3_col_id,gact3)
        plt.scatter(pfl3_col_id,gact)
        

# plt.plot(turn,color='k')
# plt.plot(turn_rough,color='r')
# da = ug.get_ang_velocity(etp.cxa.ft2['ft_heading'].to_numpy(),etp.cxa.pv2['relative_time'].to_numpy())
# plt.plot(.1*da/np.std(da),color='b')
#%%
plt.close('all')
# plt.scatter(phase,eb,c=turn_rough)
# plt.scatter(phase[np.abs(turn)<.01],eb[np.abs(turn)<.01])

etp.cxa.plot_traj_arrow_heat(['fsb_upper'],turn,cmin=-.5,cmax=.5,a_sep=5)

etp.cxa.plot_traj_arrow_heat(['fsb_upper'],turn_rough,cmin=-.5,cmax=.5,a_sep=5)
etp.cxa.plot_traj_arrow_heat(['fsb_upper'],turn,cmin=-.5,cmax=.5,a_sep=5)
#%% All wedge model
plt.close('all')
from scipy.interpolate import interpn
poffset = ug.circ_subtract(col12_theta[0],-np.pi)
eb = ug.circ_subtract(etp.cxa.pdat['phase_eb'],-poffset) # add phase offset to match anatomy
phase = ug.circ_subtract(etp.cxa.pdat['phase_fsb_upper'],-poffset)

ebw = etp.cxa.pdat['wedges_eb']



fsb = etp.cxa.pdat['wedges_fsb_upper']
x_old = np.linspace(0, 1, 16)
x_new = np.linspace(0, 1, 12)

fsb_interp = np.apply_along_axis(lambda row: np.interp(x_new, x_old, row), 1, fsb)
turn = np.zeros(len(eb))
turn_rough = np.zeros(len(eb))
for ie,e in enumerate(eb):
    act_vector = (np.cos(PFL3_input_angles-e)+1)*activation_sign
    pact = np.matmul(act_vector,PFL3_inputmatN)
    
    gact = 2*(np.cos(-phase[ie]+pfl3_thetas)+1)
    tact = np.exp(pact+gact)
    
    
    L = np.sum(tact[PFL3_LAL==-1])
    R = np.sum(tact[PFL3_LAL==1])
    turn[ie]= (R-L)/np.sum(R+L)
    
    gact2 = fsb_interp[ie,pfl3_col_id]*5
    tact = np.exp(pact+gact2)
    
    L = np.sum(tact[PFL3_LAL==-1])
    R = np.sum(tact[PFL3_LAL==1])
    turn_rough[ie]= (R-L)/np.sum(R+L)

plt.plot(turn,color='k')
plt.plot(turn_rough,color='r')
da = ug.get_ang_velocity(etp.cxa.ft2['ft_heading'].to_numpy(),etp.cxa.pv2['relative_time'].to_numpy())
plt.plot(.1*da/np.std(da),color='b')
