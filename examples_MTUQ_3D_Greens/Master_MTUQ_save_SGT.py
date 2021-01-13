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
from obspy.io.sac import SACTrace
from mtuq.event import Origin
import obspy

import scipy.io as sio
from numpy.linalg import inv

from mtuq import open_db
from mtuq.io.clients.SPECFEM3D_SAC import Client
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.process_data_NZ import ProcessData
from scipy.signal import resample
#import pdb; 

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

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

#def read_stat_name(station_file):
#
#    with open(station_file, 'r') as f:
#        lines = f.readlines()
#    line0=lines[0].split()
#    nRec=int(line0[0])
#    R=np.zeros((nRec,3))
#    statnames = [] 
#    for i in range(1,nRec+1):
#        line_i=lines[i].split()
#        R[i-1,0]=int(line_i[0])
#        R[i-1,1]=int(line_i[1])
#        R[i-1,2]=int(line_i[2])
#        statnames.append(line_i[3])
#    return nRec, R, statnames

def read_stat_name_utm(station_file):

    with open(station_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nRec=int(line0[0])
    R=np.zeros((nRec,2))
    statnames = [] 
    for i in range(1,nRec+1):
        line_i=lines[i].split()
        R[i-1,0]=float(line_i[3])
        R[i-1,1]=float(line_i[4])
        statnames.append(line_i[2])
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

#def search_stat_utm(R,statnames,statname):
#    i_rec = statnames.index(statname)
#    return i_rec
        

def write_SGT_sac(GV,G3,R,statnames,S,ishot_arr,dt):
    os.system('rm SAVE_Green_2013p544960/*')     
    for i,statname in enumerate(statnames_RGT):
        i_rec = statnames.index(statname)
        for k in range(0,3):
        #for k in range(2,3):
            fw_file='/home/andrei/workspace/GMT_tools/mtuq_src/data/INV_MarlVM_DH_2km_REAL_160km_new__PART_RGT__Kernels_SGT__Dev_Strain/'+statname+'/'+GV[k]
            _, _, _, _, _, _, _, localnp, nt, indx, coord_sgt, dh,_, _, _, _, _, _, _, _, _, Mxx, Myy, Mzz, Mxy, Mxz, Myz = read_sgt(fw_file)
            print(nt)
            print(fw_file)    
            #input('-->')  
          
            for ishot_id in range(0,len(ishot_arr)):
                origin = Origin({
                    'time': '2013-07-21T15:15:11.707Z',
                    'latitude': -41.450298,
                    'longitude': -41.450298,
                    'depth_in_m': 10000.0,
                    'id': '2013p544960'
                    }) 
                
                mkdir_p('SAVE_Green_'+origin.id)                                    
   
                
                ishot = ishot_arr[ishot_id]
                source = S[ishot-1,:] 
                #print(source)
                #print(coord_sgt)
                i_source = search_coord(source, coord_sgt, localnp)
                #print(i_source)
                #input('-->')
         
        #        AA[ishot_id,i,k,:,0]=Mxx[i_source,:]
        #        AA[ishot_id,i,k,:,4]=Myy[i_source,:]
        #        AA[ishot_id,i,k,:,8]=Mzz[i_source,:]
        ##
        #        AA[ishot_id,i,k,:,1]=Mxy[i_source,:]
        #        AA[ishot_id,i,k,:,2]=Mxz[i_source,:]
        #        AA[ishot_id,i,k,:,5]=Myz[i_source,:]
                    
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt, b=0, data=Mxx[i_source,:])
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.'+G3[k]+'.Mxx.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt, b=0, data=Myy[i_source,:])
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.'+G3[k]+'.Myy.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt, b=0, data=Mzz[i_source,:])
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.'+G3[k]+'.Mzz.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt, b=0, data=Mxy[i_source,:])
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.'+G3[k]+'.Mxy.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt, b=0, data=Mxz[i_source,:])
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.'+G3[k]+'.Mxz.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt, b=0, data=Myz[i_source,:])
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.'+G3[k]+'.Myz.sac', byteorder='little') 

                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
                               t0=0, t1=200, delta= dt, b=0, data=Mxx[i_source,:])
                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mrr.sac', byteorder='little') 

                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
                               t0=0, t1=200, delta= dt, b=0, data=Myy[i_source,:])
                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mtt.sac', byteorder='little') 

                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
                               t0=0, t1=200, delta= dt, b=0, data=Mzz[i_source,:])
                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mpp.sac', byteorder='little') 

                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
                               t0=0, t1=200, delta= dt, b=0, data=Mxy[i_source,:])
                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mrt.sac', byteorder='little') 

                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
                               t0=0, t1=200, delta= dt, b=0, data=Mxz[i_source,:])
                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mrp.sac', byteorder='little') 

                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
                               t0=0, t1=200, delta= dt, b=0, data=Myz[i_source,:])
                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mtp.sac', byteorder='little') 
                
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt/4, b=0, data= resample(Mxx[i_source,0:188],750))
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mrr.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt/4, b=0, data= resample(Myy[i_source,0:188],750))
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mtt.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt/4, b=0, data= resample(Mzz[i_source,0:188],750))
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mpp.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt/4, b=0, data= resample(Mxy[i_source,0:188],750))
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mrt.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt/4, b=0, data= resample(Mxz[i_source,0:188],750))
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mrp.sac', byteorder='little') 
#
#                syn_sac = SACTrace(kstnm=statname, kcmpnm=G3[k], stla=R[i_rec,1], stlo=R[i_rec,0], evla=origin.latitude, evlo=origin.longitude, evdp=10.0, nzyear=2000, nzjday=1, nzhour=0, nzmin=0, nzsec=0, nzmsec=0,
#                               t0=0, t1=200, delta= dt/4, b=0, data= resample(Myz[i_source,0:188],750))
#                syn_sac.write('SAVE_Green_'+origin.id+'/NZ.'+statname+'.10.'+G3[k]+'.Mtp.sac', byteorder='little') 


#                _,_, syn_sac.stats.sac['baz'] = \
#                    obspy.geodetics.gps2dist_azimuth(
#                        syn_sac.stats.sac['evla'], syn_sac.stats.sac['evlo'],
#                        syn_sac.stats.sac['stla'], syn_sac.stats.sac['stlo'])  
#                    
#                syn_sac[0].stats.back_azimuth = syn_sac.stats.sac['baz']
#                #Rotate back to NEZ    
#                syn_sac.rotate('RT->NE')    
         
    
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
#Emod3D cut
lowcut = 0.025
#highcut = 1.0
highcut = 0.1
##bw cut
#lowcut = 0.1
#highcut = 0.333
##sw cut
#lowcut = 0.05
#highcut = 0.1

fc = highcut  # Cut-off frequency of the filter
w = fc / (fs / 2) # Normalize the frequency
b, a = signal.butter(4, w, 'low')

#station_file = '../../StatInfo/STATION.txt'  
#station_file = 'STATION_dh_2km.txt'
#nRec, R, statnames = read_stat_name(station_file)
station_file = 'STATION_utm_dh_2km.txt'  
nRec, R, statnames = read_stat_name_utm(station_file)

#source_file = '../../StatInfo/SOURCE.txt'
source_file = 'SOURCE_SRF.txt.27s.2km'
#source_file = 'SOURCE_coordfile.txt'
nShot, S, sNames = read_source_new(source_file)

#statnames_RGT = ['NNZ']
statnames_RGT = ['QRZ','NNZ','WEL','THZ','KHZ','GVZ']
#statnames_RGT = statnames

#ishot_arr=np.linspace(1,nShot,nShot).astype('int')
ishot_arr=[2]#ev 2013p544960
#ishot_arr=[1,2,3,4,5]
print(sNames[2-1])

Nr=len(statnames_RGT)
Ns=len(ishot_arr)
GV=['AAA_fx.sgt','AAA_fy.sgt','AAA_fz.sgt']
#G3=['HHE','HHN','HHZ']
G3=['T','R','Z']

delta_T=5
flo=1.0
delay_Time=(3/flo)

path_weights= fullpath('data/examples/20130721151511707/weight.dat')
event_id=     '20130721151511707'
model=        'ak135'

process_bw = ProcessData(
    filter_type='Bandpass',
    freq_min= 0.1,
    freq_max= 0.333,
    pick_type='taup',
    taup_model=model,
    window_type='body_wave',
    window_length=15.,
    capuaf_file=path_weights,
    )

process_sw = ProcessData(
    filter_type='Bandpass',
    freq_min=0.05,
    freq_max=0.1,
    pick_type='taup',
    taup_model=model,
    window_type='surface_wave',
    window_length=30.,
    capuaf_file=path_weights,
    )

origin = Origin({
    'time': '2013-07-21T15:15:11.707Z',
    'latitude': -41.450298,
    'longitude': -41.450298,
    'depth_in_m': 10000.0,
    'id': '2013p544960'
    }) 
################################Store the RGT matrix
write_SGT_sac(GV,G3,R,statnames,S,ishot_arr,dt)

db = Client('SAVE_Green_2013p544960')    
    
db = open_db('SAVE_Green_2013p544960',format='SPECFEM3D')
greens = db.get_greens_tensors(stations[0],origin)


#wavelet = Trapezoid(
#    magnitude=4.5)
#
#print('Processing Greens functions...\n')
#greens.convolve(wavelet)
#greens_bw = greens.map(process_bw)
#greens_sw = greens.map(process_sw)

greens_bw = greens
greens_sw = greens



    
            
            

