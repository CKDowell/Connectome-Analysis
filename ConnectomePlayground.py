# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:32:03 2023

@author: dowel
"""
from neuprint import Client
import numpy as np
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
from neuprint import fetch_adjacencies, NeuronCriteria as NC
import pandas as pd
import matplotlib.pyplot as plt



# Objective:
    
    
    
    # 4. Find networks with top MBON input - rationalise by compartment and NT type
    # 5. Obtain candidate neurons for further study
def top_inputs(names):
    # 1. Get top inputs to HdeltaC population
    # 2. Get top inputs to FC2 neurons
    criteria = NC(type=names)
    neuron_df, conn_df = fetch_adjacencies(None, criteria)
    prenames = conn_df['bodyId_pre']
    weights = conn_df['weight']
    idlib = neuron_df['bodyId']
    typelib = neuron_df['type']
    idlib_n = pd.Series.to_numpy(idlib)
    typelib = pd.Series.to_numpy(typelib,'str')
    typelib_u = np.unique(typelib)
    t_inputs = np.empty(len(typelib_u))
    ncells = np.empty(len(typelib_u))
    weights_n = pd.Series.to_numpy(weights)
    prenames_n = pd.Series.to_numpy(prenames)
    
    for i, t in enumerate(typelib_u):
        ma_n = typelib==t
        #print(t)
        tids = idlib_n[ma_n]
        id_idx = np.in1d(prenames_n,tids)
        #print(np.sum(weights_n(ma)))
        t_inputs[i] = np.sum(weights_n[id_idx])
        ncells[i] = np.sum(ma_n)
    return typelib_u, t_inputs, ncells



## Fig params
figsize_ = [1000,1000]


# hDeltaC inputs  ##################
typelib, t_inputs, ncells = top_inputs('hDeltaC')
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
plt.show()

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
# hDeltaJ inputs ###############
typelib, t_inputs, ncells = top_inputs('hDeltaJ')
x = np.arange(0,len(typelib))
i = np.argsort(t_inputs)[::-1]

t_inputs = t_inputs[i]
typelib = typelib[i]
top_i = np.linspace(0,19,20,dtype = 'int')

tan_index = [i for i, s in enumerate(typelib) if "FB" in s]

f = plt.Figure(figsize = (6,2))
plt.plot(x[top_i],t_inputs[top_i])
plt.xticks(x[top_i],typelib[top_i],rotation=45,fontsize = 10)
plt.title('hDeltaJ inputs')
plt.show()

mn_in = t_inputs/ncells
f = plt.Figure(figsize = (6,4))

ymax = np.max(t_inputs)
plt.plot([0, ymax], [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],2), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],5), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],10), [0, ymax], color = 'k')
plt.scatter(mn_in,t_inputs)
plt.scatter(mn_in[tan_index],t_inputs[tan_index],c='g')


plt.xlabel('Mean input')
plt.ylabel('total input')
plt.title('hDeltaJ')
plt.show()
del tan_index
# FC2A inputs ##################

typelib, t_inputs, ncells = top_inputs('FC2A')
x = np.arange(0,len(typelib))
i = np.argsort(t_inputs)[::-1]

t_inputs = t_inputs[i]
typelib = typelib[i]
top_i = np.linspace(0,19,20,dtype = 'int')

tan_index = [i for i, s in enumerate(typelib) if "FB" in s]

f = plt.Figure(figsize_)
plt.plot(x[top_i],t_inputs[top_i])
plt.xticks(x[top_i],typelib[top_i],rotation=45)
plt.title('FC2A inputs')
plt.show()

mn_in = t_inputs/ncells
f = plt.Figure(figsize = (6,4))
ymax = np.max(t_inputs)
plt.plot([0, ymax], [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],2), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],5), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],10), [0, ymax], color = 'k')
plt.scatter(mn_in,t_inputs)
plt.scatter(mn_in[tan_index],t_inputs[tan_index],c='g')




plt.xlabel('Mean input')
plt.ylabel('total input')
plt.title('FC2A')
plt.show()
del tan_index
# FC2B inputs  ##################

typelib, t_inputs, ncells = top_inputs('FC2B')
x = np.arange(0,len(typelib))
i = np.argsort(t_inputs)[::-1]

t_inputs = t_inputs[i]
typelib = typelib[i]
top_i = np.linspace(0,19,20,dtype = 'int')

tan_index = [i for i, s in enumerate(typelib) if "FB" in s]

f = plt.Figure(figsize_)
plt.plot(x[top_i],t_inputs[top_i])
plt.xticks(x[top_i],typelib[top_i],rotation=45)
plt.title('FC2B inputs')
plt.show()

mn_in = t_inputs/ncells
f = plt.Figure(figsize = (6,4))
ymax = np.max(t_inputs)
plt.plot([0, ymax], [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],2), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],5), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],10), [0, ymax], color = 'k')
plt.scatter(mn_in,t_inputs)
plt.scatter(mn_in[tan_index],t_inputs[tan_index],c='g')



plt.xlabel('Mean input')
plt.ylabel('total input')
plt.title('FC2B')
plt.show()
del tan_index

# FC2 C

typelib, t_inputs, ncells = top_inputs('FC2C')
x = np.arange(0,len(typelib))
i = np.argsort(t_inputs)[::-1]

t_inputs = t_inputs[i]
typelib = typelib[i]
top_i = np.linspace(0,19,20,dtype = 'int')

tan_index = [i for i, s in enumerate(typelib) if "FB" in s]

f = plt.Figure(figsize_)
plt.plot(x[top_i],t_inputs[top_i])
plt.xticks(x[top_i],typelib[top_i],rotation=45)
plt.title('FC2C inputs')
plt.show()

mn_in = t_inputs/ncells
f = plt.Figure(figsize = (6,4))
ymax = np.max(t_inputs)
plt.plot([0, ymax], [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],2), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],5), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],10), [0, ymax], color = 'k')
plt.scatter(mn_in,t_inputs)
plt.scatter(mn_in[tan_index],t_inputs[tan_index],c='g')



plt.xlabel('Mean input')
plt.ylabel('total input')
plt.title('FC2C')
plt.show()
del tan_index

# PFL3 inputs  ##################

typelib, t_inputs, ncells = top_inputs('PFL3')
x = np.arange(0,len(typelib))
i = np.argsort(t_inputs)[::-1]

t_inputs = t_inputs[i]
typelib = typelib[i]
top_i = np.linspace(0,19,20,dtype = 'int')

tan_index = [i for i, s in enumerate(typelib) if "FB" in s]

f = plt.Figure(figsize_)
plt.plot(x[top_i],t_inputs[top_i])
plt.xticks(x[top_i],typelib[top_i],rotation=45)
plt.title('PFL3 inputs')
plt.show()

mn_in = t_inputs/ncells
f = plt.Figure(figsize = (6,4))

ymax = np.max(t_inputs)
plt.plot([0, ymax], [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],2), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],5), [0, ymax], color = 'k')
plt.plot(np.divide([0, ymax],10), [0, ymax], color = 'k')
plt.scatter(mn_in,t_inputs)
plt.scatter(mn_in[tan_index],t_inputs[tan_index],c='g')



plt.xlabel('Mean input')
plt.ylabel('total input')
plt.title('PFL3 inputs')
plt.show()

## Get FSB tangential neuron TN-TN connectivity matrix, 
# Do hiearchical clustering to get out modes

# Colour inputs by these modes


def like4like(names):
    # 3. From these inputs, compile tangential neuron 'networks'
    
    return t_nets
    

    