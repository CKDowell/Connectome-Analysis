# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:16:45 2024

@author: dowel
"""

#%%
import caveclient
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
#%%
client = caveclient.CAVEclient()
client.auth.setup_token(make_new=False)
#%%
my_token = ""
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
#%% Iterate through MBONs to get strength of lateralised connections to DNs
# This will help guide alternative hypotheses as to how odours may be navigated
types = cell_class_type_annos_df['cell_type'].to_numpy()
mbdx = [i  for i,m in enumerate(types) if 'MBON' in m ]

cids = cell_class_type_annos_df['pt_root_id'][mbdx].to_numpy()

#%%
 
DNdx = [i for i,m in enumerate(types) if 'DN' in m]
DNIds = cell_class_type_annos_df['pt_root_id'][DNdx].to_numpy()
DN_conns = np.zeros(len(cids))
for i,cd in enumerate(cids):
    print(i)
    presyn_df = client.materialize.query_view("valid_synapses_nt_np_v6", filter_in_dict={"pre_pt_root_id": [cd]})
    uid = presyn_df['post_pt_root_id'].unique()
    for u in uid:
        udx = DNIds==u
        if sum(udx)>0:
        
            DN_conns[i] = DN_conns[i]+np.sum(presyn_df['post_pt_root_id'].to_numpy()==u)
        
#%%
x = np.arange(0,len(DN_conns))
plt.plot(x,DN_conns)
plt.xticks(x,labels=cids,rotation=90)
plt.subplots_adjust(bottom=0.4)
#%% Also try and get a sense of strength of DN connections from non PFL FSB neurons
i = np.argsort(-DN_conns)
t_tab = np.zeros((len(DN_conns),2),dtype='int64')
t_tab[:,1] = cids[i]
t_tab[:,0] = DN_conns[i]


#%% Lateralised feedback loops to MB









