# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:39:09 2024

@author: dowel
"""

from neuprint import Client
import numpy as np
c = Client('neuprint.janelia.org',dataset='hemibrain:v1.2.1', token ='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRvd2VsbC5ja0BnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tQR3JQak5vclh3WXM0WWZnNld2SkV3U0N5NlVmQlhnYXlaZm5hMlpZZj1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTg3NDgxNDMwNn0.aJVCj8lW1kvChpMy8zz2_LY1RgBjw1hmLL3vTHE4nys')
c.fetch_version()
s = c.fetch_skeleton(548907426)
from neuprint import fetch_neurons, fetch_adjacencies, NeuronCriteria as NC
import matplotlib.pyplot as plt 




n,n2 = fetch_neurons(NC(type='FB4X'))
s = c.fetch_skeleton(548907426)


# Plot skeleton segments (in 2D)
x = s['x'].to_numpy()
y = s['y'].to_numpy()
z = s['z'].to_numpy()
r = s['rowId'].to_numpy()
l = s['link'].to_numpy()
fbdx = np.logical_and(y<26000,x>18500)
x = x[fbdx]
y = y[fbdx]
z = z[fbdx]
l = l[fbdx]
r = r[fbdx]
plt.figure()
for p in range(len(x)-1):
    t_x = np.append(x[p+1], x[r==l[p+1]])
    t_y = np.append(z[p+1], z[r==l[p+1]])
    plt.plot(t_x,-t_y, color='b') 


s = c.fetch_skeleton(1131840673)


# Plot skeleton segments (in 2D)
x = s['x'].to_numpy()
y = s['y'].to_numpy()
z = s['z'].to_numpy()
r = s['rowId'].to_numpy()
l = s['link'].to_numpy()
fbdx = np.logical_and(y<26000,x>18500)
x = x[fbdx]
y = y[fbdx]
z = z[fbdx]
l = l[fbdx]
r = r[fbdx]
for p in range(len(x)-1):
    t_x = np.append(x[p+1], x[r==l[p+1]])
    t_y = np.append(z[p+1], z[r==l[p+1]])
    plt.plot(t_x,-t_y, color='r') 
    
ax = plt.gca()
ax.set_aspect('equal')