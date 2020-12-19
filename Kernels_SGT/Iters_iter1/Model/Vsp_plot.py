#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:22:45 2019

@author: user
"""

#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np

import matplotlib.pyplot as plt
#import pdb; 
def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp

# #############################
nx=65
ny=65
nz=40
snap_file1='vs3dfile_opt.s'
snap_file2='vp3dfile_opt.p'
snap_file3='rho3dfile_opt.d'
Vs=Vsp_read(nx,ny,nz,snap_file1)
Vp=Vsp_read(nx,ny,nz,snap_file2)
rho=Vsp_read(nx,ny,nz,snap_file3)

print('max Vs =',str(np.max(Vs)))
print('min Vs =',str(np.min(Vs)))
print('max Vp =',str(np.max(Vp)))
print('min Vp =',str(np.min(Vp)))

K=Vs[30,:,:]
fig = plt.figure(figsize=(6, 4))
plt.imshow(K)
plt.show()    
