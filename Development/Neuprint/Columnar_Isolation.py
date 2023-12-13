# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:58:28 2023

@author: dowel
"""

#%% Point of this analysis is to investigate the isolation of columnar neurons 
# from input derived from the instantaneous heading direction system
#%% 
# To do
# 1. Define heading inputs
# 2. Take proportion of all synaptic inputs
# 3. 

#%% 
from Stable.ConnectomeFunctions import  top_inputs, top_outputs
from Stable.ConnectomeFunctions import defined_in_out
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import seaborn as sns
#%% 
hdeltas = ['hDeltaA', 'hDeltaB', 'hDeltaC', 'hDeltaD', 'hDeltaE', 'hDeltaF', 
           'hDeltaG', 'hDeltaH', 'hDeltaI', 'hDeltaJ', 'hDeltaK', 'hDeltaL',
           'hDeltaM','FC2A','FC2B','FC2C']
superclasses = ['FB','hDelta','ExR','EPG','ER','FS','FC','PEN','PEG','PFL','PFN','PFR','vDelta']
superclass_dx = [2,    1,      0   ,  0  ,  0 , 1  , 1  , 0   ,  0  ,  0  ,  0  ,  0  ,  1]
in_sum1 = np.empty([len(hdeltas), len(superclasses)+1],dtype = float)
in_sum2 = np.zeros([len(hdeltas),4],dtype=float)


for i, h in enumerate(hdeltas):
    print(h)
    typelib_u, t_inputs, ncells = top_inputs(h)
    total_in = np.sum(t_inputs)
    for r, s in enumerate(superclasses):
        idx = [it for it, tl in enumerate(typelib_u) if s in tl]
        in_sum1[i,r] = np.sum(t_inputs[idx])/total_in
        in_sum2[i,superclass_dx[r]] = in_sum2[i,superclass_dx[r]]+in_sum1[i,r]
        
in_sum1[:,-1] = 1- np.sum(in_sum1[:,:-1],axis=1)
in_sum2[:,-1] = 1- np.sum(in_sum2[:,:-1],axis=1)
#%%
savedir = "Y:\Data\Connectome\Connectome Mining\hDeltaIN_OUT"
plt.close('all')
df = pd.DataFrame(in_sum2,columns=['Heading','Processed_Col','Tangential','Other'])

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks",rc=custom_params)  # Optional: Set the style

# Transpose the DataFrame for better plotting
# Plot the stacked bar chart
ax = df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xticks(np.linspace(0,len(hdeltas)-1,len(hdeltas)),labels=hdeltas)
plt.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()
savename = os.path.join(savedir,'SimpleInputs.png')
plt.savefig(savename)

superclasses2 = ['FB','hDelta','ExR','EPG','ER','FS','FC','PEN','PEG','PFL','PFN','PFR','vDelta','Other']
df  = df = pd.DataFrame(in_sum1,columns=superclasses2)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks",rc=custom_params)  # Optional: Set the style

# Transpose the DataFrame for better plotting
# Plot the stacked bar chart
ax = df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xticks(np.linspace(0,len(hdeltas)-1,len(hdeltas)),labels=hdeltas)
plt.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()
plt.figure()
plt.imshow(np.transpose(in_sum1),vmin=0,vmax=0.25,interpolation='none',aspect='auto')
plt.xticks(np.linspace(0,len(hdeltas)-1,len(hdeltas)),labels=hdeltas,rotation=90)
plt.subplots_adjust(bottom=0.2)
plt.yticks(np.linspace(0,len(superclasses2)-1,len(superclasses2)),labels=superclasses2)
plt.ylabel('Input neuron class')
plt.colorbar()
plt.show()
savename = os.path.join(savedir,'FullInputs.png')
plt.savefig(savename)
#%% 
hdeltas = ['hDeltaA', 'hDeltaB', 'hDeltaC', 'hDeltaD', 'hDeltaE', 'hDeltaF', 
           'hDeltaG', 'hDeltaH', 'hDeltaI', 'hDeltaJ', 'hDeltaK', 'hDeltaL',
           'hDeltaM','FC2A','FC2B','FC2C']
superclasses = ['FB','hDelta','ExR','EPG','ER','FS','FC','PEN','PEG','PFL','PFN','PFR','vDelta']
superclass_dx = [2,    1,      0   ,  0  ,  0 , 1  , 1  , 0   ,  0  ,  0  ,  0  ,  0  ,  1]
in_sum1 = np.empty([len(hdeltas), len(superclasses)+1],dtype = float)
in_sum2 = np.zeros([len(hdeltas),4],dtype=float)


for i, h in enumerate(hdeltas):
    print(h)
    typelib_u, t_inputs, ncells = top_outputs(h)
    total_in = np.sum(t_inputs)
    for r, s in enumerate(superclasses):
        idx = [it for it, tl in enumerate(typelib_u) if s in tl]
        in_sum1[i,r] = np.sum(t_inputs[idx])/total_in
        in_sum2[i,superclass_dx[r]] = in_sum2[i,superclass_dx[r]]+in_sum1[i,r]
        
in_sum1[:,-1] = 1- np.sum(in_sum1[:,:-1],axis=1)
in_sum2[:,-1] = 1- np.sum(in_sum2[:,:-1],axis=1)
#%%
plt.close('all')
df = pd.DataFrame(in_sum2,columns=['Heading','Processed_Col','Tangential','Other'])

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks",rc=custom_params)  # Optional: Set the style

# Transpose the DataFrame for better plotting
# Plot the stacked bar chart
ax = df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xticks(np.linspace(0,len(hdeltas)-1,len(hdeltas)),labels=hdeltas)
plt.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()
superclasses2 = ['FB','hDelta','ExR','EPG','ER','FS','FC','PEN','PEG','PFL','PFN','PFR','vDelta','Other']
df  = df = pd.DataFrame(in_sum1,columns=superclasses2)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks",rc=custom_params)  # Optional: Set the style

# Transpose the DataFrame for better plotting
# Plot the stacked bar chart
ax = df.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xticks(np.linspace(0,len(hdeltas)-1,len(hdeltas)),labels=hdeltas)
plt.subplots_adjust(bottom=0.2)

# Show the plot
plt.show()
plt.figure()
plt.imshow(np.transpose(in_sum1),vmin=0,vmax=0.25,interpolation='none',aspect='auto')
plt.xticks(np.linspace(0,len(hdeltas)-1,len(hdeltas)),labels=hdeltas,rotation=90)
plt.subplots_adjust(bottom=0.2)
plt.yticks(np.linspace(0,len(superclasses2)-1,len(superclasses2)),labels=superclasses2)
plt.ylabel('Output neuron class')
plt.colorbar()
plt.show()
#%%
out_dict = defined_in_out(hdeltas,hdeltas)
#%%
plt.close('all')
plt.imshow(out_dict['con_mat'],vmin=0,vmax=150)
plt.xticks(np.linspace(0,len(hdeltas)-1,len(hdeltas)),labels=out_dict['in_types'],rotation=90)
plt.yticks(np.linspace(0,len(hdeltas)-1,len(hdeltas)),labels=out_dict['in_types'])
plt.subplots_adjust(bottom=0.2)
plt.colorbar()
plt.ylabel('Pre synapse')
plt.xlabel('Post synpase')
savename = os.path.join(savedir,'hDelta_FC_conmat.png')
plt.savefig(savename)