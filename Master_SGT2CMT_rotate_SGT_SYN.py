#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
#import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import time
from scipy import fftpack
from scipy import signal
from qcore import timeseries
from scipy import integrate
from scipy.signal import butter, lfilter
from scipy.signal import resample

import scipy.io as sio
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
    return data_shift

def read_sgt(matrix_file_fx):
    open_file = open(matrix_file_fx, "rb")
    geoproj = np.fromfile(open_file, dtype="i4", count=1)
    [modellon, modellat, modelrot, xshift, yshift] = np.fromfile(open_file, dtype="f4", count=5)
    [globnp, localnp, nt] = np.fromfile(open_file, dtype="i4", count=3)
    #print(localnp) 
    #print(globnp)
    #input('-->')    
    indx = np.zeros((localnp,1))
    coord_sgt = np.zeros((localnp,3))
    
    sgtheader_index = np.zeros((localnp,1))
    sgtheader_geoproj = np.zeros((localnp,1))    
    sgtheader_5floats = np.zeros((localnp,5))    
    sgtheader_nt = np.zeros((localnp,1))   
    sgtheader_7floats = np.zeros((localnp,7))       
    sgtheader_3src_coords = np.zeros((localnp,3))
    sgtheader_3floats = np.zeros((localnp,3))        
    sgtheader_3record_coords = np.zeros((localnp,3))    
    sgtheader_72floats = np.zeros((localnp,7))    
    
    Mxx = np.zeros((localnp,nt))
    Myy = np.zeros((localnp,nt))
    Mzz = np.zeros((localnp,nt))
    
    Mxy = np.zeros((localnp,nt))
    Mxz = np.zeros((localnp,nt))
    Myz = np.zeros((localnp,nt))
       
    
    for ik in range(0,localnp):
        indx[ik] = np.fromfile(open_file, dtype="i8", count=1)
        coord_sgt[ik,:] = np.fromfile(open_file, dtype="i4", count=3)
        dh = np.fromfile(open_file, dtype="f4", count=1)
        #print(coord_sgt[ik,:])
        #input('-->') 
     
    for ik in range(0,localnp):
        
        #struct sgtheader114    #/* sgt header for v1.14 */
        #   {
        #   long long indx;  /* index of this SGT */
        #   int geoproj;     /* =0: RWG local flat earth; =1: RWG great circle arcs; =2: UTM */
        #   float modellon;  /* longitude of geographic origin */
        #   float modellat;  /* latitude of geographic origin */
        #   float modelrot;  /* rotation of y-axis from south (clockwise positive)   */
        #   float xshift;    /* xshift of cartesian origin from geographic origin */
        #   float yshift;    /* yshift of cartesian origin from geographic origin */
        #   int nt;          /* number of time points                                */
        #   float xazim;     /* azimuth of X-axis in FD model (clockwise from north) */
        #   float dt;        /* time sampling                                        */
        #   float tst;       /* start time of 1st point in GF                        */
        #   float h;         /* grid spacing                                         */
        #   float src_lat;   /* site latitude */
        #   float src_lon;   /* site longitude */
        #   float src_dep;   /* site depth */
        #   int xsrc;        /* x grid location for source (station in recip. exp.)  */
        #   int ysrc;        /* y grid location for source (station in recip. exp.)  */
        #   int zsrc;        /* z grid location for source (station in recip. exp.)  */
        #   float sgt_lat;   /* SGT location latitude */
        #   float sgt_lon;   /* SGT location longitude */
        #   float sgt_dep;   /* SGT location depth */
        #   int xsgt;        /* x grid location for output (source in recip. exp.)   */
        #   int ysgt;        /* y grid location for output (source in recip. exp.)   */
        #   int zsgt;        /* z grid location for output (source in recip. exp.)   */
        #   float cdist;     /* straight-line distance btw site and SGT location */
        #   float lam;       /* lambda [in dyne/(cm*cm)] at output point             */
        #   float mu;        /* rigidity [in dyne/(cm*cm)] at output point           */
        #   float rho;       /* density [in gm/(cm*cm*cm)] at output point           */
        #   float xmom;      /* moment strength of x-oriented force in this run      */
        #   float ymom;      /* moment strength of y-oriented force in this run      */
        #   float zmom;      /* moment strength of z-oriented force in this run      */
        #   };      
        sgtheader_index[ik] = np.fromfile(open_file, dtype="i8", count=1)
        sgtheader_geoproj[ik] = np.fromfile(open_file, dtype="i4", count=1)
        
#        [modellon_sgt, modellat_sgt, modelrot_sgt, xshift, yshift] = np.fromfile(open_file, dtype="f4", count=5)
        sgtheader_5floats[ik,:] = np.fromfile(open_file, dtype="f4", count=5)       
        
        sgtheader_nt[ik] = np.fromfile(open_file, dtype="i4", count=1)
        
#        [xazim, dt, tst, h, src_lat, src_lon, src_dep] = np.fromfile(open_file, dtype="f4", count=7)
        sgtheader_7floats[ik,:] = np.fromfile(open_file, dtype="f4", count=7)          
        #print(sgtheader_7floats[ik,0])
        #input('-->')         
#        [xsrc, ysrc, zsrc] = np.fromfile(open_file, dtype="i4", count=3) 
        sgtheader_3src_coords[ik,:] = np.fromfile(open_file, dtype="i4", count=3)
        
#        [sgt_lat, sgt_lon, sgt_dep] = np.fromfile(open_file, dtype="i4", count=3) 
        sgtheader_3floats[ik,:] = np.fromfile(open_file, dtype="f4", count=3)        

#        [xsgt, ysgt, zsgt] = np.fromfile(open_file, dtype="i4", count=3) 
        sgtheader_3record_coords[ik,:] = np.fromfile(open_file, dtype="i4", count=3)
        
#        [cdist, lam, mu, rho, xmom, ymom, zmom] = np.fromfile(open_file, dtype="f4", count=3)
        sgtheader_72floats[ik,:] = np.fromfile(open_file, dtype="f4", count=7)
        #print(sgtheader_72floats[ik,:])
        #input('-->') 
        Mxx[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)
        Myy[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)
        Mzz[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)         
        
        Mxy[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)
        Mxz[ik,:] = np.fromfile(open_file, dtype="f4", count=nt)
        Myz[ik,:] = np.fromfile(open_file, dtype="f4", count=nt) 
        #print([max(Mxx[ik,:]), max(Myy[ik,:]), max(Mzz[ik,:])])
        
    return geoproj, modellon, modellat, modelrot, xshift, yshift, globnp, localnp, nt, indx, coord_sgt, dh, sgtheader_index, sgtheader_geoproj, sgtheader_5floats, sgtheader_nt, sgtheader_7floats, \
           sgtheader_3src_coords, sgtheader_3floats, sgtheader_3record_coords, sgtheader_72floats, Mxx, Myy, Mzz, Mxy, Mxz, Myz    
   
def search_coord(source, coord_sgt, n_coord):
    i_source=-1
    for ik in range(0, n_coord):
        #print(np.sum(np.square(np.array(coord_sgt[ik,:])-np.array(source))))
        if (np.sum(np.square(np.array(coord_sgt[ik,:])-np.array(source))))<3:
            i_source=ik
            break
            
    if(i_source==-1):
        print('no source in the list')
    
    return i_source
        
    
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

fs = 1/dt
#lowcut = 0.005
lowcut = 0.025
#highcut = 0.05
#highcut = 0.1
#highcut = 0.2
highcut = 1.0

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

#station_file = '../../StatInfo/STATION.txt'  
station_file = '../../StatInfo/STATION_dh_2km.txt'
nRec, R, statnames = read_stat_name(station_file)

#source_file = '../../StatInfo/SOURCE.txt'
source_file = '../../StatInfo/SOURCE_SRF.txt.27s.2km'
#source_file = 'SOURCE_coordfile.txt'
nShot, S, sNames = read_source_new(source_file)

#statnames_RGT = ['NNZ']
#statnames_RGT = ['NNZ','QRZ','WEL']
statnames_RGT = statnames

#ishot_arr=np.linspace(1,nShot,nShot).astype('int')
ishot_arr=[1]
#ishot_arr=[1,2,3,4,5]
print(sNames[1-1])

Nr=len(statnames_RGT)
Ns=len(ishot_arr)
GV=['AAA_fx.sgt','AAA_fy.sgt','AAA_fz.sgt']

delta_T=5
flo=1.0
delay_Time=(3/flo)
################################Store the RGT matrix
#AA=np.zeros([Ns,Nr,3,nt0,6])
AA=np.zeros([Ns,Nr,3,nts,9])
#ij_array=[0, 1, 2, 4, 5, 8]    
M_all=np.zeros([Ns,9])

for i,statname in enumerate(statnames_RGT):
    for k in range(0,3):
    #for k in range(2,3):
        fw_file='Dev_Strain/'+statname+'/'+GV[k]
        _, _, _, _, _, _, _, localnp, nt, indx, coord_sgt, dh,_, _, _, _, _, _, _, _, _, Mxx, Myy, Mzz, Mxy, Mxz, Myz = read_sgt(fw_file)
        print(nt)
        print(fw_file)    
        #input('-->')        
        for ishot_id in range(0,len(ishot_arr)):
            ishot = ishot_arr[ishot_id]
            source = S[ishot-1,:] 
            #print(source)
            #print(coord_sgt)
            i_source = search_coord(source, coord_sgt, localnp)
            #print(i_source)
            #input('-->')
 
            AA[ishot_id,i,k,:,0]=Mxx[i_source,:]
            AA[ishot_id,i,k,:,4]=Myy[i_source,:]
            AA[ishot_id,i,k,:,8]=Mzz[i_source,:]
#
            AA[ishot_id,i,k,:,1]=Mxy[i_source,:]
            AA[ishot_id,i,k,:,2]=Mxz[i_source,:]
            AA[ishot_id,i,k,:,5]=Myz[i_source,:]

#            AA[ishot_id,i,k,:,0] = time_shift_emod3d(Mxx[i_source,:],delay_Time,dt)
#            AA[ishot_id,i,k,:,4] = time_shift_emod3d(Myy[i_source,:],delay_Time,dt)
#            AA[ishot_id,i,k,:,8] = time_shift_emod3d(Mzz[i_source,:],delay_Time,dt)
#
#            AA[ishot_id,i,k,:,1] = time_shift_emod3d(Mxy[i_source,:],delay_Time,dt)

#            AA[ishot_id,i,k,:,2] = time_shift_emod3d(Mxz[i_source,:],delay_Time,dt)
#
#            AA[ishot_id,i,k,:,5] = time_shift_emod3d(Myz[i_source,:],delay_Time,dt)
            
#            AA[ishot_id,i,k,:,0] = butter_bandpass_filter(AA[ishot_id,i,k,:,0], lowcut, highcut, fs, order=4)
#            AA[ishot_id,i,k,:,4] = butter_bandpass_filter(AA[ishot_id,i,k,:,4], lowcut, highcut, fs, order=4)
#            AA[ishot_id,i,k,:,8] = butter_bandpass_filter(AA[ishot_id,i,k,:,8], lowcut, highcut, fs, order=4)

#            AA[ishot_id,i,k,:,1] = butter_bandpass_filter(AA[ishot_id,i,k,:,1], lowcut, highcut, fs, order=4)
            AA[ishot_id,i,k,:,3] = AA[ishot_id,i,k,:,1]

#            AA[ishot_id,i,k,:,2] = butter_bandpass_filter(AA[ishot_id,i,k,:,2], lowcut, highcut, fs, order=4)
            AA[ishot_id,i,k,:,6] = AA[ishot_id,i,k,:,2]           

#            AA[ishot_id,i,k,:,5] = butter_bandpass_filter(AA[ishot_id,i,k,:,5], lowcut, highcut, fs, order=4)
            AA[ishot_id,i,k,:,7] = AA[ishot_id,i,k,:,5]

#print(AA)                                          
np.savetxt('AA_SGT.txt',AA.reshape(-1))
#input('-->')
##############################################

#R_all_arr=np.loadtxt('../../Kernels/index_all_ncc_pyflex.txt')
#R_all=R_all_arr.reshape([nRec,3,nShot])
R_all=np.ones([nRec,3,nShot])

GV=['.090','.000','.ver']

for ishot_id in range(0,len(ishot_arr)):
    ishot=ishot_arr[ishot_id]
    mainfolder='../../Kernels/Vel_opt/Vel_ob_'+str(ishot)+'/'
#    mainfolder='../../Kernels/Vel_ob_ref/Vel_ob_'+str(ishot)+'/'
#    mainfolder_o='../../Kernels/Vel_ob_new_V2/Vel_ob_'+str(ishot)+'/'
    mainfolder_o='../../Kernels/Vel_ob_200s_HH_13s/Vel_ob_'+str(ishot)+'/'
    
    obs=np.zeros(int(3*Nr*nt))
    #print(3*Nr*nt)
    AA_ishot=AA[ishot_id,:,:,:,:]
    AA_ishot=np.reshape(AA_ishot,[Nr*3*nt,9])
    AA_max=100*np.max(np.abs(AA_ishot))
#    ik=0 
#    for i,statname in enumerate(statnames):
    for ik,statname in enumerate(statnames_RGT):
#        if((statname in statnames_RGT) and (R_all[i,k,ishot-1]==1)):
        if((statname in statnames) and (R_all[statnames.index(statname),k,ishot-1]==1)):
            i = statnames.index(statname)
            #print(sum(R_all[i,:,ishot-1]))

            for k in range(0,3):
            #for k in range(1,2):
                s0=statname+GV[k]
    #            if ((distance<200) and (distance>0) and (R_all[i,k,ishot-1]==1)):
    
                stat_data_0_S_org  = timeseries.read_ascii(mainfolder+s0)
                stat_data_0_S = time_shift_emod3d(stat_data_0_S_org,delay_Time,dt)
                #stat_data_0_S  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_S)
    
                #stat_data_0_O  = timeseries.read_ascii(mainfolder_o+s0)
                #stat_data_0_O  = np.multiply(signal.tukey(int(num_pts),0.1),stat_data_0_O)
    
                stat_data_0_O  = stat_data_0_S
 
                #stat_data_0_S = butter_bandpass_filter(stat_data_0_S, lowcut, highcut, fs, order=4)
                #stat_data_0_O = butter_bandpass_filter(stat_data_0_O, lowcut, highcut, fs, order=4)
    
                ##flexwin
                #e_s_c_name = str(ishot)+'.'+s0+'.win'
                #filename = '../../Kernels/ALL_WINs_pyflex_temp/'+e_s_c_name
                #    
                #t_on, t_off, td_shift, cc = read_flexwin(filename)
                #tx_on=int(t_on/dt)
                #tx_off=int(t_off/dt)

                #wd=np.zeros(stat_data_0_S.shape)    
                #wd[tx_on:tx_off+1]=1     
    
                #stat_data_0_S = np.multiply(stat_data_0_S,wd)
                #stat_data_0_O = np.multiply(stat_data_0_O,wd) 
                
                #stat_data_0_S = np.cumsum(stat_data_0_S)*dt          
                #stat_data_0_O = np.cumsum(stat_data_0_O)*dt                     

#                obs[nt*k*ik+nt*k:nt*k*ik+nt*(k+1)] = stat_data_0_O
                
#                obs[nt0*k*ik+nt0*k:nt0*k*ik+nt0*(k+1)]=resample(stat_data_0_S,int(nt0))
                print(nt*3*ik+nt*k)
                #if(k<1):
                #    obs[nt*3*ik+nt*k:nt*3*ik+nt*(k+1)] = stat_data_0_O
                #else:
                #    obs[nt*3*ik+nt*k:nt*3*ik+nt*(k+1)] = stat_data_0_O
                obs[nt*3*ik+nt*k:nt*3*ik+nt*(k+1)] = stat_data_0_O
#            ik=ik+1
    sio.savemat('Ad_new.mat', {'d': obs, 'AA0': AA_ishot})
    np.savetxt('AA_ishot.txt',AA_ishot.reshape(-1))
    np.savetxt('obs.txt',obs.reshape(-1))    
    input('-->')                                        
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

print(M_all)    
M_all_array=M_all.reshape(-1)
np.savetxt('CMT_inv_13events_noconv.txt',M_all_array)
    
            
            

