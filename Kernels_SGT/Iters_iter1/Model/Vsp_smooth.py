# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:54:45 2019

@author: user
"""

#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import scipy.ndimage as ndimage

#import matplotlib.pyplot as plt
#import pdb; 
def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp
    
def write_model_Vsp(rho,Vs,Vp):
    [ny,nz,nx]=Vs.shape

    model_file='rho3dfile_smooth_2km.d'
    fid=open(model_file,'wb')
    sek1 = np.array(rho, dtype=np.float32)
    sek1.astype('float32').tofile(fid)

    model_file0='vs3dfile_smooth_2km.s'
    model_file1='vp3dfile_smooth_2km.p'

    fid00=open(model_file0,'wb')
    sek2 = np.array(Vs, dtype=np.float32)
    sek2.astype('float32').tofile(fid00)

    fid01=open(model_file1,'wb')
    sek3 = np.array(Vp, dtype=np.float32)
    sek3.astype('float32').tofile(fid01)
    

# #############################
nx=65
ny=65
nz=40

snap_file1='vs3dfile_h_2km.s'
snap_file2='vp3dfile_h_2km.p'
snap_file3='rho3dfile_h_2km.d'

Vs=Vsp_read(nx,ny,nz,snap_file1)
Vp=Vsp_read(nx,ny,nz,snap_file2)
rho=Vsp_read(nx,ny,nz,snap_file3)

Vs=ndimage.gaussian_filter(Vs, 3)
Vp=ndimage.gaussian_filter(Vp, 3)

write_model_Vsp(rho,Vs,Vp)



