#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:00:02 2019

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import os
#nx=50;ny=200;nz=75; 
#nts=2500;
#nt=floor(nts/20)+1;
#dx=0.4;
#dy=dx;
#dz=dx;

def write_model_Vsp_name(rho,Vs,Vp,fname):
    [ny,nz,nx]=Vs.shape
    #[nx,nz,ny]=Vs.shape
    
    model_file='rho3d'+fname+'.d'
    fid=open(model_file,'wb')
    sek1 = np.array(rho, dtype=np.float32)
    sek1.astype('float32').tofile(fid)
    
    model_file0='vs3d'+fname+'.s'
    model_file1='vp3d'+fname+'.p'

    fid00=open(model_file0,'wb')
    sek2 = np.array(Vs, dtype=np.float32)
    sek2.astype('float32').tofile(fid00)

    fid01=open(model_file1,'wb')
    sek3 = np.array(Vp, dtype=np.float32)
    sek3.astype('float32').tofile(fid01)
    
def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp    

def read_stat_name(station_file):

    with open(station_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nRec=int(line0[0])
    R=np.zeros((nRec,3))
    statnames = [] 
    for i in range(1,nRec+1):
        line_i=lines[i].split()
        R[i-1,0]=int(line_i[0])
        R[i-1,1]=int(line_i[1])
        R[i-1,2]=int(line_i[2])
        statnames.append(line_i[3])
    return nRec, R, statnames

def read_source(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
    
    return nShot, S        
            
########################
nx=65
ny=65
nz=40
#nz0=21

#nx=100
#ny=100
#nz=75

dx=2
dy=dx
dz=dx

x=dx*np.arange(1,nx+1,1)-dx
y=dy*np.arange(1,ny+1,1)-dy
z=dz*np.arange(1,nz+1,1)-dz

snap_file1='vs3dfile.s'
snap_file2='vp3dfile.p'
snap_file3='rho3dfile.d'

Vs0=Vsp_read(nx,ny,nz,snap_file1)
Vp0=Vsp_read(nx,ny,nz,snap_file2)
rho0=Vsp_read(nx,ny,nz,snap_file3)

Vs=3.0*np.ones((ny,nz,nx))
Vp=6.0*np.ones((ny,nz,nx))
rho=np.max(rho0)*np.ones((ny,nz,nx))

#Vs[0:ny,0:nz0,0:nx]=Vs0
#Vp[0:ny,0:nz0,0:nx]=Vp0
#rho[0:ny,0:nz0,0:nx]=rho0

#fname_true='file'
fname_init='file_homogeneous_2km'
#
#Vs=2.0*np.ones((ny,nz,nx))
#Vp=4.0*np.ones((ny,nz,nx))
#rho=2.7*np.ones((ny,nz,nx))
write_model_Vsp_name(rho,Vs,Vp,fname_init)

