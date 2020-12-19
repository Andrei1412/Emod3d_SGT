#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy import fftpack
from scipy import signal
from qcore import timeseries
from scipy import integrate
from scipy.signal import butter, lfilter
from scipy.signal import resample

from numpy.linalg import inv
#import pdb; 
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_srf(fname):
    """
    Convinience function for reading files in the Graves and Pitarka format
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    data = []

    for line in lines[7:]:
        data.append([float(val) for val in line.split()])

    data=np.concatenate(data) 
    
    line5=lines[5].split()
    line6=lines[6].split()   
#    num_pts=float(line1[0])
    dt_srf=float(line5[7])
    strike=float(line5[4])
    dip=float(line5[5])
    rake=float(line6[0])

    return data, dt_srf, strike, dip, rake

def conv_source(eii3_xs,source,nt,dt):
    """
         Convolve original RGT with source s(t)
    """
#    (nt,M) = eii3_xs.shape
#    nt=nv+1
    Gii= np.zeros((nt))
    for it in range(0,nt):
        Gii[it]=Gii[it]+np.multiply(eii3_xs[it],source[nt-it-1])            
#        Gii[it-1]=np.multiply(eii3_xs[it-1],source[nt-it])     
#        Gii[it-1]=np.multiply(eii3_xs[it-1],source[it-1])
    return Gii

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    #print(sek.shape)
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

def read_source_new(source_file):
    
    with open(source_file, 'r') as f:
        lines = f.readlines()    
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    sNames=[]
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
        sNames.append(line_i[3])
    
    return nShot, S, sNames    

def time_shift_emod3d(data,delay_Time,dt):
    n_pts = len(data)
    ndelay_Time = int(delay_Time/(dt))
    data_shift = np.zeros(data.shape)
    data_shift[0:n_pts-ndelay_Time] = data[ndelay_Time:n_pts]
    return data_shift
   
# Read_Dev_strain_ex2.m
nx=65
ny=65
#nz=75
nz=40
nts=2000
nt0=400
nxyz=nx*ny*nz

num_pts=2000; dt=0.08;
t = np.arange(num_pts)*dt

fs = 1/dt
lowcut = 0.05
#highcut = 0.4
highcut = 0.1

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

station_file = '../../StatInfo/STATION.txt'  
nRec, R, statnames = read_stat_name(station_file)

source_file = '../../StatInfo/SOURCE.txt'
nShot, S, sNames = read_source_new(source_file)

statname='CBGS'
with open('Dev_Strain/'+statname+'/RGT_X/fwd01_xyzts.exx', 'rb') as fid:
#with open('/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/fwd01_xyzts.exx', 'rb') as fid:
    data_array = np.fromfile(fid, np.float32)
    data_array=data_array[0:14]
    dx=data_array[8]
    dy=data_array[9]
    dz=data_array[10]
    dt0=data_array[11]
    del data_array
    
#snap_file1='../../Model/Models/vs3dfile_h.s';
#snap_file2='../../Model/Models/vp3dfile_h.p';
#snap_file3='../../Model/Models/rho3dfile.d';
#snap_file1='/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/vs3dfile_h.s';
#snap_file2='/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/vp3dfile_h.p';    
#snap_file3='/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/rho3dfile_h.d';
#Vs=Vsp_read(nx,ny,nz,snap_file1)
#Vp=Vsp_read(nx,ny,nz,snap_file2)
#rho=Vsp_read(nx,ny,nz,snap_file3)

#Lamda=np.multiply(rho,(np.multiply(Vp,Vp)-2*np.multiply(Vs,Vs)));
#Mu=np.multiply(rho,np.multiply(Vs,Vs))
#Kappa=Lamda+2/3*Mu

#statnames_RGT = ['CBGS','CMHS','DFHS','NNBS','PPHS','REHS']
#statnames_RGT = ['CACS', 'CBGS', 'CRLZ', 'REHS', 'NNBS']
statnames_RGT = ['CBGS', 'CRLZ', 'NNBS']
#eij_names = ['exx','eyy','ezz','exy','exz','eyz']
eij_names = ['exx','exy','exz','eyy','eyz','ezz']
#eij_names = ['exx','exx','exx','exx','exx','exx']
#fw_file='Dev_Strain/'+statname+'/RGT_X/fwd01_xyzts.';

ishot_arr=[1, 5, 28, 37, 55, 98, 120, 69]
Nr=len(statnames_RGT)
Ns=len(ishot_arr)
GV=['RGT_X','RGT_Y','RGT_Z']
#ishot_arr=[45, 69, 132, 142, 143]
#source=np.zeros(nt0)
source_all=np.zeros([Ns,nt0])
M_all=np.zeros([Ns,9])

for ishot_id in range(0,len(ishot_arr)):
    ishot=ishot_arr[ishot_id]
    #os.system('cp /home/user/workspace/GMPlots/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')  
    os.system('cp /scale_wlg_nobackup/filesets/nobackup/nesi00213/RunFolder/tdn27/rgraves/NZVMs/Christchurch_Events/Srf/'+str(sNames[ishot-1])+'.srf srf_file.srf')
    fname='srf_file.srf'
    data, dt_srf, strike, dip, rake = read_srf(fname)
    #resample the srf time series to the full simulation time
    N_sample=int(dt/dt_srf);
    data_dt = resample(data,int(len(data)/N_sample))
    data_dt_new = np.zeros(num_pts)
    data_dt_new[0:len(data_dt)] = data_dt
    data_dt_new = butter_bandpass_filter(data_dt_new, lowcut, highcut, fs, order=4)
    source_all[ishot_id,:] = resample(data_dt_new,nt0)
#    resample(stat_data_0_S,int(nt0/num_pts))    
################################Store the RGT matrix
#AA=np.zeros([Ns,Nr,3,nt0,6])
AA=np.zeros([Ns,Nr,3,nt0,9])
ij_array=[0, 1, 2, 4, 5, 8]    

for i,statname in enumerate(statnames_RGT):
    for ij, eij_name in enumerate(eij_names):
        for k in range(0,3):
            fw_file='Dev_Strain/'+statname+'/'+GV[k]+'/fwd01_xyzts.'
            matrix_file_fw=fw_file+eij_name
#            matrix_file_fw='/home/user/workspace/GMPlots/synth_VMs/Medium_VMs/fwd01_xyzts.exx'
            fid1=open(matrix_file_fw, 'rb')
            matrix_dummy_fw=np.fromfile(fid1, np.float32)
            matrix_dummy_fw=matrix_dummy_fw[15:]
            
            for ishot_id in range(0,len(ishot_arr)):
                ishot = ishot_arr[ishot_id]
                source = source_all[ishot_id,:] 
                          
                eii3 = np.reshape(matrix_dummy_fw,[nt0,ny,nz,nx]) 
        
                eii3_fw=eii3[:,int(S[ishot-1,1])-1,int(S[ishot-1,2])-1,int(S[ishot-1,0])-1]
                
#                AA[ishot_id,i,k,:,int(ij_array[ij])]=conv_source(eii3_fw,source,nt0,dt0)
                AA[ishot_id,i,k,:,int(ij_array[ij])]=eii3_fw
                
                if(int(ij_array[ij])==1):
#                    AA[ishot_id,i,k,:,3]=conv_source(eii3_fw,source,nt0,dt0)
                    AA[ishot_id,i,k,:,3]=eii3_fw
                if(int(ij_array[ij])==2):
                    AA[ishot_id,i,k,:,6]=eii3_fw     
                if(int(ij_array[ij])==5):
                    AA[ishot_id,i,k,:,7]=eii3_fw                      
                    
np.savetxt('AA_8events_new.txt',AA.reshape(-1))
input('-->')
##############################################
#R_all_arr=np.loadtxt('../../index_all_ncc_pyflex_gt05_V2_excluded.txt')
#R_all=R_all_arr.reshape([nRec,3,nShot])

#R_Time_record_arr = np.loadtxt('R_Time_record_148s_dh_2km_num_pts_V2.txt') 
R_Time_record_arr = np.loadtxt('R_Time_record_148s_dh_2km.txt') 
R_Time_record = R_Time_record_arr.reshape([2,nShot,nRec])

GV=['.090','.000','.ver']

delta_T=5
flo=0.1
delay_Time=(3/flo)

for ishot_id in range(0,len(ishot_arr)):
    ishot=ishot_arr[ishot_id]
    
#    os.system('cp Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_ob/')
#    os.system('cp Vel_ob_ref_01Hz/Vel_ob_'+str(ishot)+'/*.* /home/user/workspace/GMPlots/Sim/Vel_es/')    

#    mainfolder='/home/user/workspace/GMPlots/Sim/Vel_es/'
#    mainfolder_o='/home/user/workspace/GMPlots/Sim/Vel_ob/'

    #mainfolder='Vel_opt/Vel_ob_'+str(ishot)+'/'
    mainfolder='../../Kernels/Vel_ob_ref/Vel_ob_'+str(ishot)+'/'
#    mainfolder_o='../../Kernels/Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/'
    mainfolder_o='../../Kernels/Vel_ob_ref/Vel_ob_'+str(ishot)+'/'
    
    obs=np.zeros(int(3*Nr*nt0))
    AA_ishot=AA[ishot_id,:,:,:,:]
    AA_ishot=np.reshape(AA_ishot,[Nr*3*nt0,9])
    AA_max=100*np.max(np.abs(AA_ishot))
    
#    b = np.array([1, 1, 1, -1, 1, 1, -1, -1, 1])
   
    ################################
#    R_ishot_arr=np.loadtxt('Dump/R_ishot_'+str(ishot)+'.txt')
#    if ((S[ishot-1,2]<20) and (np.sum(R_all[:,:,ishot-1])>3)):
    
#    for i,statname in enumerate(statnames):
    for i,statname in enumerate(statnames_RGT):
        ik=i
        if(statname in statnames_RGT):
            #distance=((R[i,1]-S[ishot-1,1])**2+(R[i,2]-S[ishot-1,2])**2+(R[i,0]-S[ishot-1,0])**2)**(0.5)
    
            for k in range(0,3):
                s0=statname+GV[k]
    #            if ((distance<200) and (distance>0) and (R_all[i,k,ishot-1]==1)):
    
                stat_data_0_S_org  = timeseries.read_ascii(mainfolder+s0)
                stat_data_0_S = time_shift_emod3d(stat_data_0_S_org,delay_Time,dt)
                stat_data_0_S  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_S)
    
                stat_data_0_O  = timeseries.read_ascii(mainfolder_o+s0)
                stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_O)
                #stat_data_0_S = signal.detrend(stat_data_0_S)             
                #stat_data_0_O = signal.detrend(stat_data_0_O)
    
        #        stat_data_0_S = signal.filtfilt(b, a, stat_data_0_S)           
        #        stat_data_0_O = signal.filtfilt(b, a, stat_data_0_O)
    
                stat_data_0_S = butter_bandpass_filter(stat_data_0_S, lowcut, highcut, fs, order=4)
                stat_data_0_O = butter_bandpass_filter(stat_data_0_O, lowcut, highcut, fs, order=4)
    
                wd=np.zeros(stat_data_0_S.shape)    
                wd[int(R_Time_record[0,ishot-1,i]/dt):int(R_Time_record[1,ishot-1,i]/dt)+1]=1     
    
                stat_data_0_S = np.multiply(stat_data_0_S,wd)
                stat_data_0_O = np.multiply(stat_data_0_O,wd) 
                
                stat_data_0_S = np.cumsum(stat_data_0_S)*dt          
                stat_data_0_O = np.cumsum(stat_data_0_O)*dt                     
                
#                obs[nt0*k*ik+nt0*k:nt0*k*ik+nt0*(k+1)]=resample(stat_data_0_S,int(nt0)) 
                obs[nt0*k*ik+nt0*k:nt0*k*ik+nt0*(k+1)]=-resample(stat_data_0_S,int(nt0))

                            
    #Add iso(M0)=0 to A and d
#    d=np.r_[d, 0]
    obs=np.r_[obs,[0,0,0,0]]
    b = np.zeros((1,9)); b[0,0] = 1; b[0,4] = 1; b[0,8] = 1;  b = b*AA_max;
    AA_ishot = np.r_[AA_ishot, b]    
    #Add symmetriy constraint
#    d=np.r_[d, 0]; d=np.r_[d, 0]; d=np.r_[d, 0];
    
    b = np.zeros((1,9)); b[0,1] = 1; b[0,3] = -1; b = b*AA_max;
    AA_ishot = np.r_[AA_ishot, b] 

    b = np.zeros((1,9)); b[0,2] = 1; b[0,6] = -1; b = b*AA_max;
    AA_ishot = np.r_[AA_ishot, b] 

    b = np.zeros((1,9)); b[0,5] = 1; b[0,7] = -1; b = b*AA_max;
    AA_ishot = np.r_[AA_ishot, b]     
    #Inversion
    AA_inv=inv(np.matmul(np.transpose(AA_ishot),AA_ishot))     
    AA_d=np.matmul(np.transpose(AA_ishot),obs)
    M_ishot=np.matmul(AA_inv,AA_d)
    M_all[ishot_id,:]=M_ishot
    
M_all_array=M_all.reshape(-1)
np.savetxt('CMT_inv_8events_noconv.txt',M_all_array)
    
            
            

