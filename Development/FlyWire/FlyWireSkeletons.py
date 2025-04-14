# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 13:45:04 2025

@author: dowel
"""

#skeleton_dir = r'D:\ConnectomeData\FlywireWholeBrain\flywire_skeleton_swcs2_uncom\swcs'
skeleton_dir = r'D:\ConnectomeData\FlywireWholeBrain\sk_lod1_783_healed_unzip'
import os
import numpy as np
import neurom as nm
import navis
from matplotlib import pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 
#%% FB6H
colours = np.array( [[166,206,227],
[31,120,180],
[178,223,138],
[51,160,44],
[251,154,153],
[227,26,28],
[253,191,111],
[255,127,0],
[202,178,214],
[106,61,154],
[255,255,153],
[177,89,40]])/255
cell_IDs = [720575940613052200,720575940630840493]
i = 0
fi,ax =  plt.subplots()
view = ('x','-y')
for i in cell_IDs:
    swc_dir = os.path.join(skeleton_dir,str(i)+'.swc')
    neuron = navis.read_swc(swc_dir)
    
    neuron.plot2d(color=colours[0,:],view=view,ax=ax,soma=False,alpha=0.5)

# FB7B
cell_IDs = [720575940642476448,720575940626612254]
for i in cell_IDs:
    swc_dir = os.path.join(skeleton_dir,str(i)+'.swc')
    neuron = navis.read_swc(swc_dir)
    
    neuron.plot2d(color=colours[1,:],view=view,ax=ax,soma=False,alpha=0.5)

# FB5H
cell_IDs = [720575940625787536, 720575940604451505]
for i in cell_IDs:
    swc_dir = os.path.join(skeleton_dir,str(i)+'.swc')
    neuron = navis.read_swc(swc_dir)
    
    neuron.plot2d(color=colours[2,:],view=view,ax=ax,soma=False,alpha=0.5)
#FB4M
cell_IDs = [720575940623807672,720575940623358099,720575940608140380,720575940640039040]
for i in cell_IDs:
    swc_dir = os.path.join(skeleton_dir,str(i)+'.swc')
    neuron = navis.read_swc(swc_dir)
    
    neuron.plot2d(color=colours[3,:],view=view,ax=ax,soma=False,alpha=0.5)
#FB4L
cell_IDs = [720575940622760682,
            720575940619104027,
            720575940608312924,
           720575940634338515
           ]
for i in cell_IDs:
    swc_dir = os.path.join(skeleton_dir,str(i)+'.swc')
    neuron = navis.read_swc(swc_dir)
    
    neuron.plot2d(color=colours[4,:],view=view,ax=ax,soma=False,alpha=0.5)

#FB2A
cell_IDs = [720575940640002256,
            720575940621782381,720575940628705271,720575940625875402,720575940626314061,720575940636924479]

for i in cell_IDs:
    swc_dir = os.path.join(skeleton_dir,str(i)+'.swc')
    neuron = navis.read_swc(swc_dir)
    
    neuron.plot2d(color=colours[5,:],view=view,ax=ax,soma=False,alpha=0.5)
# plt.xlim([370000,710000])
# plt.ylim([70000,300000])
ax.set_aspect('equal', adjustable='box')
plt.savefig(os.path.join(r'Y:\Presentations\2025\04_LabMeeting\DANs','DAN_Skeletons.pdf'))
#%% hDeltaC
cell_IDs = [720575940638250112,720575940629468739,720575940628907524,
            720575940614966214,720575940624215245,720575940631366483,
            720575940623781268,720575940631725011,720575940628219479,
            720575940617494232,720575940635678942,720575940614154786,
            720575940643703588,720575940612851367,720575940620058537,
            720575940632052010,720575940628211244,720575940631983277,
            720575940622811116,720575940618694638,720575940621347583]
fi,ax =  plt.subplots()
view = ('x','-y')
for i in cell_IDs:
    swc_dir = os.path.join(skeleton_dir,str(i)+'.swc')
    neuron = navis.read_swc(swc_dir)
    
    neuron.plot2d(color='k',linewidth=0.5,view=view,ax=ax,soma=False,alpha=0.5)
ax.set_aspect('equal', adjustable='box')
plt.savefig(os.path.join(r'Y:\Presentations\2025\04_LabMeeting\hDeltaC','hDeltaC_Skeletons.pdf'))
    
    
    