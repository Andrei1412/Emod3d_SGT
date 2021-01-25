#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
#import math
import scipy.ndimage as ndimage
#import matplotlib.pyplot as plt
#import pdb; 
def read_source(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=int(line_i[0])
        S[i-1,1]=int(line_i[1])
        S[i-1,2]=int(line_i[2])
    
    return nShot, S

def write_hessian_Vsp(Hessian_S,Hessian_P):
    [ny,nz,nx]=Hessian_S.shape

    hessian_file0='Hessian_S.s'
    hessian_file1='Hessian_P.p'

    fid00=open(hessian_file0,'wb')
    sek2 = np.array(Hessian_S, dtype=np.float32)
    sek2.astype('float32').tofile(fid00)

    fid01=open(hessian_file1,'wb')
    sek3 = np.array(Hessian_S, dtype=np.float32)
    sek3.astype('float32').tofile(fid01)

def threshole_hessian(Hessian, THRESHOLD_HESS):
    [ny,nz,nx] = Hessian.shape
    Inv_Hessian = np.zeros([ny,nz,nx])
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):    
                if (Hessian[j,k,i]>THRESHOLD_HESS):
                    Inv_Hessian[j,k,i] = 1/Hessian[j,k,i]
                else:
                    Inv_Hessian[j,k,i] = 1/THRESHOLD_HESS
                    
    return   Inv_Hessian                  

nx=88
ny=88
nz=60
nxyz=nx*ny*nz

#source_file='../../../StatInfo/SOURCE_checker.txt'
#station_file='../../../StatInfo/STATION_checker.txt'
source_file='../../../StatInfo/SOURCE_SRF.txt'
station_file='../../../StatInfo/STATION_dh_4km.txt'

nShot, S = read_source(source_file)
#nRec, R = read_source(station_file)                       

##########################################################

Gra_S=np.zeros((ny,nz,nx))
Gra_P=np.zeros((ny,nz,nx))

Hessian_S=np.zeros((ny,nz,nx))
Hessian_P=np.zeros((ny,nz,nx))

#for ii in range(1,nShot+1):
ishot_arr=np.linspace(1,nShot,nShot).astype('int')
#ishot_arr=[1,2,3,4,5,6,7,8,9,10,12,13]

for ishot_id in range(0,len(ishot_arr)):
    ii=ishot_arr[ishot_id]

    print('Sum Gradient Shot '+str(ii))  
    GS_file='All_shots/GS_shot'+str(ii)+'.txt'    
    GP_file='All_shots/GP_shot'+str(ii)+'.txt'
    
    GS_arr=np.loadtxt(GS_file)
    GS=np.reshape(GS_arr,[ny,nz,nx])
    GP_arr=np.loadtxt(GP_file)
    GP=np.reshape(GP_arr,[ny,nz,nx])
    
    print('Sum Hessian Shot '+str(ii))  
    HS_file='All_shots/HS_shot'+str(ii)+'.txt'    
    HP_file='All_shots/HP_shot'+str(ii)+'.txt'
    
    HS_arr=np.loadtxt(HS_file)
    HS=np.reshape(HS_arr,[ny,nz,nx])
    HP_arr=np.loadtxt(HP_file)
    HP=np.reshape(HP_arr,[ny,nz,nx])    
    
#        #Tp=precondition_matrix(nx,ny,nz,S[ii-1,:],R,R_nf)
#        tape_file='Dump/Tp_shot_'+str(ii)+'.txt'
#        Tp_arr=np.loadtxt(tape_file)
#        Tp=np.reshape(Tp_arr,[ny,nz,nx])
    Tp=np.ones((ny,nz,nx)) 
#        
    Gra_S=Gra_S+np.multiply(GS,Tp)
    Gra_P=Gra_P+np.multiply(GP,Tp)
    
    Hessian_S=Hessian_S+np.abs(np.multiply(HS,Tp))
    Hessian_P=Hessian_P+np.abs(np.multiply(HP,Tp))       

##Tape boundary
tape_file='Dump/Tp_xyz.txt'
Tp_arr=np.loadtxt(tape_file)
Tp=np.reshape(Tp_arr,[ny,nz,nx]) 
#
Gra_S=np.multiply(Gra_S,Tp)
Gra_P=np.multiply(Gra_P,Tp)

Hessian_S=np.multiply(Hessian_S,Tp)
Hessian_P=np.multiply(Hessian_P,Tp)
          
#max_Hessian_S = np.max(Hessian_S)
#max_Hessian_P = np.max(Hessian_P)
Gra_S[:,0,:]=0
Gra_P[:,0,:]=0

Hessian_S[:,0,:]=0
Hessian_P[:,0,:]=0

Hessian_S = Hessian_S/np.max(Hessian_S)
Hessian_P = Hessian_P/np.max(Hessian_P)

THRESHOLD_HESS = 5.e-4
#
Inv_Hessian_S = threshole_hessian(Hessian_S, THRESHOLD_HESS)
Inv_Hessian_P = threshole_hessian(Hessian_P, THRESHOLD_HESS)
#
Gra_S=np.multiply(Gra_S,Inv_Hessian_S)
Gra_P=np.multiply(Gra_P,Inv_Hessian_P)
#
Gra_S=-ndimage.gaussian_filter(Gra_S, [2,1,2])
Gra_P=-ndimage.gaussian_filter(Gra_P, [2,1,2])
#
Gra_S_arr=Gra_S.reshape(-1)
np.savetxt('Gra_S.txt', Gra_S_arr)
#    
Gra_P_arr=Gra_P.reshape(-1)
np.savetxt('Gra_P.txt', Gra_P_arr)
#print('finish Summing Gradients')

#print('max Gs='+str(np.max(Gra_S_arr)))
#print('max Gp='+str(np.max(Gra_P_arr)))

Hessian_S=ndimage.gaussian_filter(Hessian_S,[2,1,2])
Hessian_P=ndimage.gaussian_filter(Hessian_P,[2,1,2])
#save hessian to binary:
write_hessian_Vsp(Hessian_S,Hessian_P) 

#Hessian_S_arr=Hessian_S.reshape(-1)
#np.savetxt('Hessian_S.txt', Hessian_S_arr)
    
#Hessian_P_arr=Hessian_P.reshape(-1)
#np.savetxt('Hessian_P.txt', Hessian_S_arr)
print('finish Summing Gradients')

#print('max Hs='+str(np.max(Hessian_S_arr)))
#print('max Hp='+str(np.max(Hessian_P_arr)))

