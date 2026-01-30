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

#%% PFL3 neurons
dx = df_meta['hemibrain_type']=='PFL3'
print(np.sum(dx))
PFL3 = df_meta['root_id'][dx].to_numpy()
presyn_PFL3 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": PFL3})
postsyn_PFL3 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": PFL3})

#%% Get PFL FSB columnar arrangement
from scipy import stats
fsbdx = postsyn_PFL3['neuropil']=='FB'
pfl3coords = np.zeros((len(PFL3),3))
pfl3_projmean = np.zeros((len(PFL3),3))
pfl3_coords_all = postsyn_PFL3['post_pt_position'].to_numpy()
pfl3_coords_all = np.array(pfl3_coords_all.tolist())



pca = PCA(n_components=3)
pfl3fit = (pfl3_coords_all-np.mean(pfl3_coords_all,axis=0))/np.std(pfl3_coords_all,axis=0)
pca.fit(pfl3fit)
b = pca.components_
proj = np.matmul(pfl3fit,pca.components_.T)
    
for i,p in enumerate(PFL3):
    pdx = postsyn_PFL3['post_pt_root_id']==p
    dx = np.logical_and(pdx,fsbdx)
    tcoords = postsyn_PFL3['post_pt_position'][dx]
    pfl3coords[i,:] = np.mean(tcoords,axis=0)
    pfl3_projmean[i,:] = np.mean(proj[dx,:],axis=0)
    
km = KMeans(n_clusters=12,n_init=50).fit(pfl3coords) # Does not always give appropriate coords
cents = km.cluster_centers_
cr = np.argsort(cents[:,0])
cents_ranked = cents[cr,:]
col_array= np.arange(0,12)
col_array2 = col_array.copy()
col_array2[cr] = col_array.copy()
pfl3_col_id = col_array2[km.labels_] 

# Re rank PB gloms 
pb_rank_cond = np.append(np.arange(0,8),np.arange(0,8))
col8_theta = np.zeros(8)
for i in range(8):
    col8_theta[i] = stats.circmean(glom_theta[pb_rank_cond==i],low=-np.pi,high=np.pi)
    
# This shift if zeroed produces a perfect PFL3 output... needto check...
col8_theta_shift = np.roll(col8_theta,0) # shift by one column since middle glom innervates right FSB


col12_theta = ug.circ_interp(np.linspace(0,7,12),np.arange(0,8),col8_theta_shift)

pfl3_thetas = col12_theta[pfl3_col_id]
    
    
    
    
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
for i,p in enumerate(PFL3):
    pdx = postsyn_PFL3['post_pt_root_id']==p
    ROIs = np.unique(postsyn_PFL3['neuropil'][pdx])
    if sum(ROIs=='LAL_L')==1:
        PFL3_LAL[i] = -1 
    elif sum(ROIs=='LAL_R')==1:
        PFL3_LAL[i] = 1
    for ie,e in enumerate(ranked_EPG_final):
        edx = postsyn_PFL3['pre_pt_root_id']==e
        dx = np.logical_and(pdx,edx)
        PFL3_inputmat[ie,i] = np.sum(dx)
        
    for ie,e in enumerate(delta7):
        edx = postsyn_PFL3['pre_pt_root_id']==e
        dx = np.logical_and(pdx,edx)
        PFL3_inputmat[ie+len(ranked_EPG_final),i] = np.sum(dx)

#%% Model output for different EB offsets without FSB input
#plt.close('all')
offsets = np.linspace(-np.pi,np.pi,100)
diff = np.zeros(len(offsets))
PFL3_inputmatN = PFL3_inputmat/np.sum(PFL3_inputmat,axis=0)
for io,o in enumerate(offsets):
    act_vector = (np.cos(PFL3_input_angles+o)+1)*activation_sign
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
plt.xlabel('Bump position (rads)')

#%% Model output for different EB offsets without FSB input
#plt.close('all')
offsets = np.linspace(-np.pi,np.pi,100)
diff = np.zeros(len(offsets))
PFL3_inputmatN = PFL3_inputmat.copy()
d7_epg_rat =np.sum(PFL3_inputmat[:len(ranked_EPG_final),:]) /np.sum(PFL3_inputmat[:])
PFL3_inputmatN[:len(ranked_EPG_final),:] = d7_epg_rat*PFL3_inputmatN[:len(ranked_EPG_final),:]/np.sum(PFL3_inputmatN[:len(ranked_EPG_final),:],axis=0) 
PFL3_inputmatN[len(ranked_EPG_final):,:] = (1-d7_epg_rat)*PFL3_inputmatN[len(ranked_EPG_final):,:]/np.sum(PFL3_inputmatN[len(ranked_EPG_final):,:],axis=0) 

for io,o in enumerate(offsets):
    act_vector = (np.cos(PFL3_input_angles+o)+1)*activation_sign
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
        act_vector = (np.cos(PFL3_input_angles+o)+1)*activation_sign
        pact = np.matmul(act_vector,PFL3_inputmatN)
        
        for i2,o2 in enumerate(offsetsG):
            gact =gstrength*(np.cos(pfl3_thetas+o2)+1)
            tact = np.exp(pact+gact)
            L = np.sum(tact[PFL3_LAL==-1])
            R = np.sum(tact[PFL3_LAL==1])
            diff[io,i2] = (R-L)/np.sum(R+L)
            #print(L,R)
    
    plt.subplot(3,3,ig+1)
    plt.imshow(np.flipud(diff),aspect='auto',cmap='coolwarm')#,vmin=-.5,vmax=.5)
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
            gact =gstrength*(np.exp(np.cos(pfl3_thetas-o2))+1)+5
            tact = np.exp(pact+gact)
            L = np.sum(tact[PFL3_LAL==-1])
            R = np.sum(tact[PFL3_LAL==1])
            diff[io,i2] = (R-L)/np.sum(R+L)
            #print(L,R)
    plt.figure(101)
    plt.subplot(3,3,ig+1)
    plt.imshow(np.flipud(diff),aspect='auto',cmap='coolwarm',vmin=-.1,vmax=.1)
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
for ig,gstrength in enumerate(gstrengths):
    gact =gstrength*(np.cos(pfl3_thetas)+1)+5
    plt.scatter(pfl3_thetas,gact,label=ig)
plt.ylim([0,15])
plt.legend()
#%% Play in some activity to the network
from analysis_funs.CX_analysis_col import CX_a
from EdgeTrackingOriginal.ETpap_plots.ET_paper import ET_paper

datadir = "Y:\Data\FCI\Hedwig\FC2_maimon2\\241106\\f1\\Trial2"
etp = ET_paper(datadir)
#%%
plt.close('all')
from scipy.interpolate import interpn
poffset = ug.circ_subtract(col12_theta[0],-np.pi)
eb = ug.circ_subtract(etp.cxa.pdat['phase_eb'],-poffset) # add phase offset to match anatomy
phase = ug.circ_subtract(etp.cxa.pdat['phase_fsb_upper'],-poffset)
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
    
    
    R = np.sum(tact[PFL3_LAL==-1])
    L = np.sum(tact[PFL3_LAL==1])
    turn[ie]= (R-L)/np.sum(R+L)
    
    gact2 = fsb_interp[ie,pfl3_col_id]*5
    tact = np.exp(pact+gact2)
    
    R = np.sum(tact[PFL3_LAL==-1])
    L = np.sum(tact[PFL3_LAL==1])
    turn_rough[ie]= (R-L)/np.sum(R+L)
    if np.mod(ie,500)==0:
        plt.figure()
        plt.scatter(pfl3_thetas,gact,color='k')
        plt.scatter(pfl3_thetas,gact2,color='r')
        plt.title(phase[ie])
        
plt.figure()
plt.scatter(phase,eb,c=turn)

plt.figure()
plt.scatter(phase,eb,c=turn_rough)
plt.figure()
plt.scatter(etp.cxa.pdat['offset_eb_phase'].to_numpy(),etp.cxa.pdat['offset_fsb_upper_phase'].to_numpy(),c=turn_rough)

#%%
etp.cxa.plot_traj_arrow_heat(['fsb_upper'],turn,cmin=-.5,cmax=.5,a_sep=5)

# plt.scatter(phase,eb,c=turn_rough)
# plt.scatter(phase[np.abs(turn)<.01],eb[np.abs(turn)<.01])
plt.plot(turn,color='k')
plt.plot(turn_rough,color='r')
etp.cxa.plot_traj_arrow_heat(['fsb_upper'],turn,cmin=-.5,cmax=.5,a_sep=5)

etp.cxa.plot_traj_arrow_heat(['fsb_upper'],turn_rough,cmin=-.5,cmax=.5,a_sep=5)
