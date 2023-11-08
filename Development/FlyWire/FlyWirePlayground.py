# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:24:21 2023

@author: dowel
"""

#%% Plan

# 1) Recapitulate the eigen experiment of the Pillow lab
# 2) Find eigen vectors with neurons of interest - FSB, MBON etc
# 3) Look at angles between eigenvectors to predict how activity spreads/interacts
# 4) Compare these eigenvector interactions to simulations of activity

#%%
datafolder = 'D:\\ConnectomeData\\FlywireWholeBrain'

class fw:
    def __init__(self):
        self.datafolder = 'D:\\ConnectomeData\\FlywireWholeBrain'
    
    def make_conmatrix(self):
        # Function takes the connectivity table and outputs a connectivity matrix
        # with synapse signs
    
    def eigen_domposition(self):
        # Function does eigen decomposition of connectivity matrix
    
    def neuprint_to_FW(self,NPneurons):
        # Function gives flywire names to neuprint neuron types
        
    def eigen_list_NP(self,NPneurons):
        # Function gives the list of egeinvectors for which a given neuron contributes from neuprint name
        
    
    
