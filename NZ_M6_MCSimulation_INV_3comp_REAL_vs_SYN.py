#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:29:10 2020

@author: andrei
"""

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

from itertools import permutations
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

def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    #print(sek.shape)
    Vp=np.reshape(sek,[ny,nz,nx])

    return Vp

def read_flexwin(filename):

    with open(filename, 'r') as f:
        lines = f.readlines()

    if(len(lines)>2):
        line2=lines[2].split()
        t_on=float(line2[1])
        t_off=float(line2[2])
        td_shift=float(line2[3])
        cc=float(line2[4])

    else:
        t_on = 0; t_off = 0; td_shift = 0; cc = 0;

    return t_on, t_off, td_shift, cc

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
#    data_shift[ndelay_Time:n_pts]  = data[0:n_pts-ndelay_Time]   
    return data_shift

#def sdr2cmt(strike, dip, rake, Mo):
#
#    #Assume that rot=0 or Mxx=Mnn otherwise Mxx = Mnn*cosA*cosA + Mee*sinA*sinA + Mne*sin2A;
#
#    PI = np.pi*1/180.0 # Convert from degree to radian
#    Mzz=np.sin(2*dip*PI)*np.sin(rake*PI)
#    Mxx=np.sin(dip*PI)*np.cos(rake*PI)*np.sin(2*strike*PI) + np.sin(2*dip*PI)*np.sin(rake*PI)*np.sin(strike*PI)**2
#    Mxx=-Mxx
#
#    Myy=np.sin(dip*PI)*np.cos(rake*PI)*np.sin(2*strike*PI) - np.sin(2*dip*PI)*np.sin(rake*PI)*np.cos(strike*PI)**2
#    Mxz=np.cos(dip*PI)*np.cos(rake*PI)*np.cos(strike*PI) + np.cos(2*dip*PI)*np.sin(rake*PI)*np.sin(strike*PI)
#    Mxz=-Mxz
#
#    Myz=np.cos(dip*PI)*np.cos(rake*PI)*np.sin(strike*PI) - np.cos(2*dip*PI)*np.sin(rake*PI)*np.cos(strike*PI)
#    Mxy=np.sin(dip*PI)*np.cos(rake*PI)*np.cos(2*strike*PI) + 0.5*np.sin(2*dip*PI)*np.sin(rake*PI)*np.sin(2*strike*PI)
#    Myz=-Myz
#
#    M9 = Mo*np.array([[Mxx, Mxy, Mxz],[Mxy, Myy, Myz],[Mxz, Myz, Mzz]])
#
#    return M9
    
#def sdr2cmt(strike, dip, rake):
#    
##    strike = 90-strike
#
#    PI = np.pi*1/180.0 # Convert from degree to radian
#    #sin2D*sinL;sin2D = two*sinD*cosD; D=dip*PI; L=rake*PI
#    Mzz=np.sin(2*dip*PI)*np.sin(rake*PI) 
#    #    
#    Mxx=np.sin(dip*PI)*np.cos(rake*PI)*np.sin(2*strike*PI) + np.sin(2*dip*PI)*np.sin(rake*PI)*np.sin(strike*PI)**2
#    Mxx=-Mxx
#
#    Myy=np.sin(dip*PI)*np.cos(rake*PI)*np.sin(2*strike*PI) - np.sin(2*dip*PI)*np.sin(rake*PI)*np.cos(strike*PI)**2
#    Mxz=np.cos(dip*PI)*np.cos(rake*PI)*np.cos(strike*PI) + np.cos(2*dip*PI)*np.sin(rake*PI)*np.sin(strike*PI)
#    Mxz=-Mxz
#
#    Myz=np.cos(dip*PI)*np.cos(rake*PI)*np.sin(strike*PI) - np.cos(2*dip*PI)*np.sin(rake*PI)*np.cos(strike*PI)
#    Mxy=np.sin(dip*PI)*np.cos(rake*PI)*np.cos(2*strike*PI) + 0.5*np.sin(2*dip*PI)*np.sin(rake*PI)*np.sin(2*strike*PI)
#    Myz=-Myz
#
#    M6 = np.array([Mxx, Myy, Mzz, Mxy, Mxz, Myz])
##    M6 = np.array([-Mxx, Myy, Mzz, Mxy, -Mxz, -Myz])    
##    M6 = np.array([Mxx, Myy, Mzz, Mxy, -Mxz, -Myz])  
##    M6 = np.array([Mxx, Myy, Mzz, Mxy, -Mxz, Myz])        
##    M6 = np.array([Mxx, Mxy, Mxz, Myy, Myz, Mzz])
#
#    return M6
    
def sdr2cmt(strike, dip, rake):
    
#    strike = 90-strike
    rperd = np.pi*1/180.0 # Convert from degree to radian
    #sin2D*sinL;sin2D = two*sinD*cosD; D=dip*PI; L=rake*PI
    arg = dip*rperd;
    cxD = np.cos(arg);
    sxD = np.sin(arg);
    
    cx2D = cxD*cxD - sxD*sxD;
    sx2D = 2*sxD*cxD;     
    
    arg = rake*rperd;
    cxL = np.cos(arg);
    sxL = np.sin(arg);
    
    arg = (strike - 90)*rperd;
    cxT = np.cos(arg);
    sxT = np.sin(arg);
    
    cx2T = cxT*cxT - sxT*sxT;
    sx2T = 2*sxT*cxT;
    
    mxx = -(sxD*cxL*sx2T + sx2D*sxL*sxT*sxT);
    myy = (sxD*cxL*sx2T - sx2D*sxL*cxT*cxT);
    mzz = sx2D*sxL;
    mxy = (sxD*cxL*cx2T + 0.5*sx2D*sxL*sx2T);
    mxz = -(cxD*cxL*cxT + cx2D*sxL*sxT);
    myz = -(cxD*cxL*sxT - cx2D*sxL*cxT);
    
    M6 = np.array([mxx, myy, mzz, mxy, mxz, myz])
#    M6 = np.array([mxx, myy, mzz, mxy, -mxz, -myz])

#    M6 = np.array([myy, mxx, mzz, mxy, myz, mxz])    
#    M6 = np.array([myy, mzz, mxx, myz, mxy, mxz])   

#    M6 = np.array([mzz, myy, mxx, myz, mxz, mxy])
#    M6 = np.array([mzz, mxx, myy, mxz, myz, mxy])    
     
    return M6
        
# Read_Dev_strain_ex2.m
nx=176
ny=176
#nz=75
nz=80
nts=2500
nt0=500
nxyz=nx*ny*nz

num_pts=2500; dt=0.08;
t = np.arange(num_pts)*dt

#tst = 2.96 #time start shift
tst = 0.05
xyz_mom = 1e+20


fs = 1/dt
lowcut = 0.025
#highcut = 0.2
highcut = 0.1

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

#station_file = '../../StatInfo/STATION.txt'  
station_file = 'STATION_dh_2km.txt'
nRec, R, statnames = read_stat_name(station_file)

#source_file = '../../StatInfo/SOURCE.txt'
source_file = 'SOURCE_SRF.txt'
#source_file = 'SOURCE_coordfile.txt'
nShot, S, sNames = read_source_new(source_file)

#statnames_RGT = ['NNZ']
#statnames_RGT = ['NNZ','QRZ','WEL']
statnames_RGT = statnames

#ishot_arr=np.linspace(1,nShot,nShot).astype('int')
ishot_arr=[1]
#ishot_arr=[5]
ishot_id = 0
ishot = ishot_arr[ishot_id]
#ishot_arr=[1,2,3,4,5]

Nr=len(statnames_RGT)
Ns=len(ishot_arr)

GV=['.090','.000','.ver']
################################Store the RGT matrix
#AA=np.zeros([Ns,Nr,3,nts,9])
AA_ishot_arr = np.loadtxt('AA_ishot.txt')

Nr=9 #statnames = statnames_shot_i
AA_ishot = AA_ishot_arr.reshape([Nr,3,nts,9])
            

    #NNZ, i=1:
i=5;#k=2;

d_org_3 = np.zeros([3,nts])
s_org_3 = np.zeros([3,nts])
d_rec_3 = np.zeros([3,nts])

for k in range(0,3): 
#    fig = plt.figure(figsize=(10,2.5))    
#    plt.plot(t , AA_ishot[i,k,:,0],label='gxx')
#    plt.plot(t , AA_ishot[i,k,:,4],label='gyy')
#    plt.plot(t , AA_ishot[i,k,:,8],label='gzz')
#    
#    plt.plot(t , AA_ishot[i,k,:,1],label='gxy')        
#    plt.plot(t , AA_ishot[i,k,:,2],label='gxz')
#    plt.plot(t , AA_ishot[i,k,:,5],label='gyz')            
#    #    plt.plot(xint, source_i_err_st[0:3,0],label=str(i))   
#    plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))   
#    plt.xlim([0,200])           
#    plt.legend(loc='best') 
#    plt.ylabel("gij",fontsize=14)
#    plt.xlabel("Time (s)",fontsize=14)
#    plt.title(sNames[ishot-1]+GV[k]+'.'+statnames[i],fontsize=14)       
#    plt.show()
#    
    #source 1, 3505099
#    strike = 69; dip = 90; rake = 151;
    Mo=2.15e+23;
    
#    strike = 165; dip = 26; rake = -95;    #    165    26   -95 / 4 stations, 0.1Hz     
    strike = 175; dip = 37; rake = 35;    #    175    37    35 / 4 stations, 0.2Hz     

    
    #source 5, 2015p013973
#    strike = 252; dip = 82; rake = 150;    
#    Mo=8.95e+22;
    
#    strike = 261; dip = 15; rake = -64;    # 261    15   -64    
#    strike = 335; dip = 54; rake = 167;    #  335    54   167       
#    strike = 334; dip = 56; rake = 169;    #  334    56   169
#    strike = 333; dip = 57; rake = 170;    #  333    57   170
    
#    strike = 342; dip = 66; rake = 3;    #    342    66     3
#    strike = 342; dip = 68; rake = 6;    #    342    66     3
#    strike = 341; dip = 67; rake = 7;    #    342    66     3
    
##    strike = 141; dip = 47; rake = -39;    #    141    47   -39 / 5 stations, 0.2Hz   
#    strike = 152; dip = 14; rake = 144;    #    152    14   144 / 3 stations, 0.1Hz      
    
    #source 12, 2017p144774
#    strike = 163; dip = 86; rake = -4;
#    Mo=1.43e+23;    
    
    #M9 = sdr2cmt(strike, dip, rake, 1)
    ###M9_arr = M9.reshape(-1) 
    #Mrr = M9[2,2]; Mtt = M9[0,0]; Mpp = M9[1,1];
    #Mrt = M9[0,2]; Mrp = -M9[1,2]; Mtp = -M9[0,1];
    #
    #M6 = [Mtt, Mpp, Mrr, -Mtp, Mrt, -Mrp]
    #print(M6)
    #AA6[i,0,:,0]
    d_arr = np.loadtxt('obs.txt')
    d= d_arr.reshape([Nr,3,nts])
    
    s_arr = np.loadtxt('syn.txt')
    s= s_arr.reshape([Nr,3,nts])    
    #
    #AA_ishot[i,k,:,1] = 0.5*AA_ishot[i,k,:,1];AA_ishot[i,k,:,2] = 0.5*AA_ishot[i,k,:,2];AA_ishot[i,k,:,5] = 0.5*AA_ishot[i,k,:,5];
    #AA_ishot[i,k,:,3] = 0.5*AA_ishot[i,k,:,1];AA_ishot[i,k,:,3] = 0.5*AA_ishot[i,k,:,6];AA_ishot[i,k,:,7] = 0.5*AA_ishot[i,k,:,7];
    M6 = sdr2cmt(strike, dip, rake)
#    print(M6)
    AA6 = np.zeros([Nr,3,nts,6])
    AA6[i,k,:,0] = AA_ishot[i,k,:,0]; AA6[i,k,:,1] = AA_ishot[i,k,:,4]; AA6[i,k,:,2] = AA_ishot[i,k,:,8];
#    AA6[i,k,:,3] = AA_ishot[i,k,:,1]; AA6[i,k,:,4] = AA_ishot[i,k,:,2]; AA6[i,k,:,5] = AA_ishot[i,k,:,5];
    AA6[i,k,:,3] = 2*AA_ishot[i,k,:,1]; AA6[i,k,:,4] = 2*AA_ishot[i,k,:,2]; AA6[i,k,:,5] = 2*AA_ishot[i,k,:,5];
    
    
    d_org = np.array(d[i,k,:])
    s_org = np.array(s[i,k,:])    
#    d_org = timeseries.read_ascii('Vel_ob_200s_HH_13s/Vel_ob_1/'+statnames[i]+GV[k])
#    if(k>0):
#        d_org = -d_org    

    d_rec = Mo/xyz_mom*(AA_ishot[i,k,:,0]*M6[0]+  AA_ishot[i,k,:,4]*M6[1]+ AA_ishot[i,k,:,8]*M6[2]+ \
    AA_ishot[i,k,:,1]*M6[3]+  AA_ishot[i,k,:,2]*M6[4]+ AA_ishot[i,k,:,5]*M6[5])
    
    #source  = timeseries.read_ascii('Stf/AAA-fx_cos1.00')
#    d_rec = time_shift_emod3d(d_rec,2.96,dt)
        
    d_org  = np.multiply(signal.tukey(int(num_pts),0.1),d_org)
    s_org  = np.multiply(signal.tukey(int(num_pts),0.1),s_org)    
    d_rec  = np.multiply(signal.tukey(int(num_pts),0.1),d_rec)
#    d_org = signal.demean(d_org)       
#    s_org = signal.demean(s_org)        
#    d_rec = signal.demean(d_rec)    
    d_org = signal.detrend(d_org)      
    s_org = signal.detrend(s_org)          
    d_rec = signal.detrend(d_rec)
    
    
    
    d_org = butter_bandpass_filter(d_org, lowcut, highcut, fs, order=4)
    s_org = butter_bandpass_filter(s_org, lowcut, highcut, fs, order=4)    
    d_rec = butter_bandpass_filter(d_rec, lowcut, highcut, fs, order=4)

    #Already rotate the observed data to account to this:
#    if(k>0):
#        d_rec = -d_rec
    
    d_org_3[k,:] = d_org
    s_org_3[k,:] = s_org    
    d_rec_3[k,:] = d_rec        
    #dn =  reshape(d,2500,10,3);
    #AAn = reshape(AA0,2500,10,3,9);
    #
    #AAn(:,:,:,2) = 0.5*AAn(:,:,:,2);AAn(:,:,:,3) = 0.5*AAn(:,:,:,3);AAn(:,:,:,6) = 0.5*AAn(:,:,:,6);
    #AAn(:,:,:,4) = 0.5*AAn(:,:,:,4);AAn(:,:,:,7) = 0.5*AAn(:,:,:,7);AAn(:,:,:,8) = 0.5*AAn(:,:,:,8);
    #dn_rec = sum(squeeze(AAn(:,9,1,:)).*m0_0(:)',2);
    
    fig1 = plt.figure(figsize=(10,2.5))
    plt.plot(t , d_org,'k',label='observed data')    
    plt.plot(t , d_rec,'r',label='reciprocal reconstruction, new CMT')
    plt.plot(t , s_org,'b',label='reciprocal reconstruction, geonet CMT')#    plt.plot(t , -d_rec,'r',label='reciprocal reconstruction')
    plt.title(sNames[ishot-1]+GV[k]+'.'+statnames[i])
    plt.xlim([0,200])
    #plt.xlim([50,100])
    #plt.xlim([25,50])
    #plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) 
    plt.legend(loc='best')
    plt.show()
    
#k=0    
#fig1 = plt.figure(figsize=(10,2.5))
#plt.plot(t , d_org_3[0],'k',label='forward simulation')
#plt.plot(t , d_rec_3[0],'r',label='reciprocal reconstruction')#    plt.plot(t , -d_rec,'r',label='reciprocal reconstruction')
#plt.title(sNames[ishot-1]+GV[k]+'.'+statnames[i])
#plt.xlim([0,200])
##plt.xlim([50,100])
##plt.xlim([25,50])
##plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0)) 
#plt.legend(loc='best')
#plt.show()    

#it=1499;
#print([AA_ishot[i,k,it,0],AA_ishot[i,k,it,4],AA_ishot[i,k,it,8],AA_ishot[i,k,it,1], AA_ishot[i,k,it,2],AA_ishot[i,k,it,5]]);    
            
            

