# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:48:56 2024

@author: dowel
"""

import caveclient
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#%%
client = caveclient.CAVEclient()
client.auth.setup_token(make_new=False)
#%%
my_token = "942d1c4bd3683cb7c09302a1d8430bb6"
client.auth.save_token(token=my_token,overwrite=True)
#%% 
datastack_name = "flywire_fafb_public"
client = caveclient.CAVEclient(datastack_name)
#%%
client.materialize.get_tables()
#%% 
NI = client.materialize.query_table("neuron_information_v2")
#%%
cell_class_type_annos_df = client.materialize.query_table("hierarchical_neuron_annotations", filter_in_dict={"classification_system": ["cell_class", "cell_type"]})
cell_class_type_annos_df
#%%
top_dir = 'D:\\ConnectomeData\\FlywireWholeBrain\\'
fn = top_dir + 'classification.csv'
df_meta = pd.read_csv(fn)

fn = top_dir + 'connections_princeton_no_threshold.csv'
df = pd.read_csv(fn)

fn = top_dir + 'neuropil_synapse_table.csv'
df_synreg = pd.read_csv(fn)
#%% Get PFL3 and PFL2 neurons
dx = df_meta['hemibrain_type']=='PFL3'
print(np.sum(dx))
PFL3s = df_meta['root_id'][dx].to_numpy()

dx = df_meta['hemibrain_type']=='PFL2'
print(np.sum(dx))
PFL2s = df_meta['root_id'][dx].to_numpy()


dx = df_meta['hemibrain_type']=='Delta7'
print(np.sum(dx))
Delta7 = df_meta['root_id'][dx].to_numpy()

dx = df_meta['hemibrain_type']=='EPG'
print(np.sum(dx))
EPG = df_meta['root_id'][dx].to_numpy()
#%% PFL3 connections
presyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": PFL3s})
postsyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": PFL3s})
#%% 
# Get LAL outputs
LAL_L = presyn_df[presyn_df['neuropil']=='LAL_L']
LAL_R = presyn_df[presyn_df['neuropil']=='LAL_R']
L_ids,Lc = np.unique(LAL_L['post_pt_root_id'].to_numpy(),return_counts=True)
R_ids,Rc = np.unique(LAL_R['post_pt_root_id'].to_numpy(),return_counts=True)
ls = np.argsort(-Lc)
Lc = Lc[ls]
L_ids = L_ids[ls]


rs = np.argsort(-Rc)
Rc = Rc[rs]
R_ids = R_ids[rs]
# Check for reciprocity

# Get outputs
synthresh = 30 
Ridsmall = R_ids[Rc>synthresh]
Lidsmall = L_ids[Lc>synthresh]

RidLid = np.append(Ridsmall,Lidsmall)
#presyn_df2 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": np.append(Ridsmall,Lidsmall)})
##postsyn_df1 = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": np.append(Ridsmall,Lidsmall)})

presyn_df1 = df[np.in1d(df['pre_pt_root_id'].to_numpy(),RidLid)]


# Get one step down
outs,oc = np.unique(presyn_df1['post_pt_root_id'].to_numpy(),return_counts=True)

oc2  =np.array([],dtype=int)
for i,o in enumerate(outs):
    dx = presyn_df1['post_pt_root_id']==o
    oc2 = np.append(oc2,np.sum(presyn_df1['syn_count'][dx]))

# ins,ic = np.unique(presyn_df1['pre_root_id'].to_numpy(),return_counts=True)
# outs2,oc2 = np.unique(presyn_df2['post_pt_root_id'].to_numpy(),return_counts=True)
# ins2,ic2 = np.unique(presyn_df2['pre_pt_root_id'].to_numpy(),return_counts=True)



outs = outs[oc2>synthresh]
total = np.append(outs,RidLid)
total = np.append(PFL3s,total)
total = np.unique(total)
#conn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": total,"post_pt_root_id":total})
#postsyn_df2= client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": total})

# Make connectivity matrix - easier to query the csvs
neurotransmitter_effects = {
    'ACH': 1,   
    'DA': -1,     
    'GABA': -1,  
    'GLUT': -1,  
    'OCT': 1,   
    'SER': 1    
    }

# In future can avoid for loop.
conmat = np.zeros((len(total),len(total)))
for i,c in enumerate(total):
    print(i)
    r1 = np.where(np.in1d(df['pre_pt_root_id'],c))[0]
    dfsmall = df[np.in1d(df['pre_pt_root_id'],c)]
    counts = dfsmall['syn_count'].to_numpy()
    nt = dfsmall['nt_type'].to_numpy()
    
    ntsign = np.zeros_like(nt)
    for n,nt in enumerate(nt):
        try :
            ntsign[n] = neurotransmitter_effects[nt]
        except:
            ntsign[n] = 1
    
    for ir,r in enumerate(r1):
        rd= np.where(total==df['post_pt_root_id'][r])[0]
        conmat[i,rd] += counts[ir]*ntsign[ir] # add in neurotransmitter types

plt.imshow(conmat,aspect='auto',interpolation='none',vmin=-30,vmax=30,cmap='coolwarm')

#%% 

total_syns = np.zeros_like(total)
for i,c in enumerate(total):
    print(i)
    total_syns[i] = df['syn_count'][df['post_pt_root_id']==c].sum()
    if total_syns[i]==0:
        print('Zero inputs!!!',i,c)
        
        
#%% Total lat
tlat = np.zeros_like(total)
for i,t in enumerate(total):
    dx = df_synreg['root_id']==t
    small_synreg = df_synreg[dx]
    lalr = small_synreg['input synapses in LAL_R'].to_numpy()
    lall = small_synreg['input synapses in LAL_L'].to_numpy()
    if lalr>0 or lall>0:
        tlat[i] = ( lalr-lall)/(lalr+lall)
#%% Propagate activity through this small network
conmat_norm = conmat.copy()
for i,t in enumerate(total_syns):
    if t==0:
        conmat_norm[:,i] = 0
    else:
        conmat_norm[:,i] = conmat_norm[:,i]/t

plt.close('all')    

# 1 Get laterality of PFL3 neurons
dx = np.in1d(df_synreg['root_id'].to_numpy(),PFL3s)
pfl3_synreg = df_synreg[dx]
lat = pfl3_synreg['output synapses in LAL_R'].to_numpy() -pfl3_synreg['output synapses in LAL_L'].to_numpy()
lat = np.sign(lat)

# 2 Stimulate all left side
left_PFL3s = pfl3_synreg['root_id'][lat<0].to_numpy()
ldx = np.in1d(total,left_PFL3s)
input_vector = np.zeros(len(total))
input_vector[ldx] = 1

right_PFL3s = pfl3_synreg['root_id'][lat>0].to_numpy()
rdx = np.in1d(total,right_PFL3s)
input_vector_r = np.zeros(len(total))
input_vector_r[rdx] = 1

# 3 Iterate through 3 times L
iter=  10
out_all_L = np.zeros((len(total),iter))
#conmat_norm = conmat/tot_syns
for i in range(iter):
    if i==0:
        out = np.matmul(input_vector,conmat)
    else:
        out = np.matmul(out_norm,conmat)
    out_all_L[:,i] = out
    out_norm = out
    out_norm[out_norm<0] = 0
    
    out_norm[out_norm>0] = 1/(1+np.exp(-out_norm[out_norm>0]/1000))
    #out_norm = out_norm+input_vector



out_all_R = np.zeros((len(total),iter))
#conmat_norm = conmat/tot_syns
for i in range(iter):
    if i==0:
        out = np.matmul(input_vector_r,conmat)
    else:
        out = np.matmul(out_norm,conmat)
    out_all_R[:,i] = out
    out_norm = out
    out_norm[out_norm<0] = 0
    
    out_norm[out_norm>0] = 1/(1+np.exp(-out_norm[out_norm>0]/1000))
    #out_norm = out_norm+input_vector_r
# 4 Look at highest activated neurons
for i in range(iter):
    plt.figure()
    x = out_all_L[:,i]
    y = out_all_R[:,i]
    mx = np.argsort(-np.abs(x))
    my = np.argsort(-np.abs(y))
    mall = np.append(mx[:10],my[:10])
    mall = np.unique(mall)
    mROI = total[mall]
    mlabs = np.array([])
    for m in mROI:
        dx = df_meta['root_id']==m
        tlab = df_meta['hemibrain_type'][dx]
        if tlab.isna().any():
            tlab = 'nan'
        mlabs = np.append(mlabs,tlab)
    plt.scatter(x,y,c=tlat,cmap='coolwarm')
    for ix,ml in  enumerate(mlabs):
        plt.text(x[mall[ix]],y[mall[ix]],ml)
        
    plt.xlabel('Left PFL3 active')
    plt.ylabel('Right PFL3 active')


#%% Get PFL3 glomerular identity
# load synapse coordinates
fn = top_dir + 'synapse_coordinates.csv'
syn_coords = pd.read_csv(fn)
postsyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"post_pt_root_id": PFL3s})

#%% Determine glomeruli ruler
all_coords = np.array([])
presyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": EPG})
#%%
pbdx = presyn_df['neuropil']=='PB'


nids = presyn_df['pre_pt_root_id'][pbdx]

pbcoords = np.array(presyn_df['pre_pt_position'][pbdx].tolist())
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
mean_coords = np.zeros((len(EPG),3))
for i,n in enumerate(EPG):
    c = i+np.zeros(np.sum(nids==n))
   # plt.scatter(pbcoords[nids==n,0],pbcoords[nids==n,1],s=5)
    cmean = np.mean(pbcoords[nids==n,:],axis=0)
    mean_coords[i,:] = cmean
    ax.scatter(cmean[0],cmean[1],cmean[2],color='k')
    #ax.scatter(pbcoords[nids==n,0],pbcoords[nids==n,1],pbcoords[nids==n,2],s=5)
    #plt.scatter3(cmean[0],cmean[1],color='k')



from sklearn.cluster import KMeans
km = KMeans(n_clusters=18).fit(mean_coords) # Does not always give appropriate coords
cents = km.cluster_centers_
cdx = np.argsort(cents[:,0])
cents = cents[cdx,:]
ax.scatter(cents[:,0],cents[:,1],cents[:,2],color='r')
#%% 
import os
from Utils.utils_general import utils_general as ug
ug.save_pick(cents,os.path.join(top_dir,'EPG_gloms.pkl'))
#%%
PFL3_out = np.append(left_PFL3s,right_PFL3s)
lat = np.append(np.zeros_like(left_PFL3s)-1,np.zeros_like(right_PFL3s)+1)

#%% Save meta data
plt.close('all')
d7_coords = np.zeros((len(PFL3_out),3))
epg_gloms = ug.load_pick(os.path.join(top_dir,'EPG_gloms.pkl'))
PFL3_gloms = np.zeros(len(PFL3_out),dtype='int')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i,p in enumerate(PFL3_out):
    pdx = postsyn_df['post_pt_root_id']==p
    pdf = postsyn_df[pdx]
    pdf = pdf[pdf['neuropil']=='PB']
    coord2 = np.array(pdf['pre_pt_position'].tolist())
   
    
    d7_coords[i,:] = np.median(coord2,axis=0)
    PFL3_gloms[i] = ug.find_nearest_euc(epg_gloms,d7_coords[i,:])
ax.scatter(d7_coords[:,0],d7_coords[:,1],d7_coords[:,2],c=PFL3_gloms,cmap='coolwarm')
ax.scatter(epg_gloms[:,0],epg_gloms[:,1],epg_gloms[:,2],color='g')
for i in range(16):
    ax.text(epg_gloms[i,0],epg_gloms[i,1],epg_gloms[i,2],i)
    
pgdx = np.argsort(PFL3_gloms)


#%% get fsb columns
fsb_coords = np.zeros((len(PFL3_out),3))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i,p in enumerate(PFL3_out):
    pdx = postsyn_df['post_pt_root_id']==p
    pdf = postsyn_df[pdx]
    pdf = pdf[pdf['neuropil']=='FB']
    coord2 = np.array(pdf['pre_pt_position'].tolist())
   
    #plt.scatter(coord2[:,0],coord2[:,1])
    fsb_coords[i,:] = np.mean(coord2,axis=0)
km = KMeans(n_clusters=12).fit(fsb_coords) # 12 columns for PFL3,more or less symmetrically sample
ax.scatter(fsb_coords[:,0],fsb_coords[:,1],fsb_coords[:,2],color='k')
ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],km.cluster_centers_[:,2],color='r')
s = np.argsort(km.cluster_centers_[:,0])
fsb_columns = km.labels_
cids = np.array([np.where(s==i)[0][0] for i in fsb_columns ])

np.where(np.in1d(s,km.labels_))[0]
#%%
#PFL3s = PFL3s[pgdx]
#PFL3_gloms = PFL3_gloms[pgdx]
#lat = lat[pgdx]
pfl_metadict = {'pbglom':PFL3_gloms,'LAL':lat,'root_id':PFL3_out,'fsb_column':cids}
ug.save_pick(pfl_metadict,os.path.join(top_dir,'PFL3_meta_data.pkl'))

