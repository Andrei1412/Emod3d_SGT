#!/usr/bin/env python
# Generated with SMOP  0.41
#from libsmop import *
import os
import numpy as np
#import os,sys
import time
#from qcore import timeseries as ts
from shared_workflow.shared import exe
from datetime import datetime
import shlex
from subprocess import Popen, PIPE
#from numpy.linalg import solve
#import math
#import matplotlib.pyplot as plt
#import pdb; 


def submit_script(script):
    res = exe("sbatch {}".format(script), debug=False)
    if len(res[1]) == 0:
        # no errors, return the job id
        return_words = res[0].split()
        job_index = return_words.index("job")
        jobid = return_words[job_index + 1]
        try:
            int(jobid)
        except ValueError:
	        print(
	            "{} is not a valid jobid. Submitting the "
	    	    "job most likely failed".format(jobid)
	        )
        return jobid

def wait_job_to_finish(jobid, time_submited):
    while True:
        #check squeue
        out_list = []
        cmd = "sacct -u tdn27 -S {} -j {} -b ".format(time_submited,jobid)
        #print(cmd)
        process = Popen(shlex.split(cmd), stdout=PIPE, encoding="utf-8")
        (output, err) = process.communicate()
        exit_code = process.wait()

        out_list.extend(filter(None, output.split("\n")[1:]))

        #print(output)
        while len(out_list) <= 1:
            time.sleep(5)
            print('re-querry sacct for jobid : {}'.format(jobid))
            out_list = []
            process = Popen(shlex.split(cmd), stdout=PIPE, encoding="utf-8")
            (output, err) = process.communicate()
            exit_code = process.wait()
            out_list.extend(filter(None, output.split("\n")[1:]))
            
###############Print out job status###################
#            print(len(out_list))
#        print(out_list)
#        print(len(out_list))
       
        job = out_list[1].split()
        if job[1] == "COMPLETED":
            return 1
        else:
#            print("waiting for job : {} to finish".format(jobid))
            time.sleep(5)
            continue

def job_submitted(job_file):
        submit_time = datetime.now().strftime('%Y-%m-%d')
        jobid = submit_script(job_file)
        wait_job_to_finish(jobid,submit_time)
###############

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

def write_slurm_SGT(statname,SLON,SLAT):
    sname='sgtrun-mpi.sl'
    #file_default='e3d_mysource.par'
    os.system('cp SGT/sgtrun-mpi_header.sl sgtrun-mpi.sl')
    fid=open(sname,'a')
    
    #set STAT = AZ_TRO
    #set SLON = -116.4257
    #set SLAT = 33.5234    
    fid.write("%s\n" %('set STAT = '+statname))
    fid.write("%s\n" %('set SLON = '+str(SLON)))
    fid.write("%s\n" %('set SLAT = '+str(SLAT)))

    with open('SGT/sgtrun-mpi_default.sl', 'r') as f:
        lines = f.readlines()
        
    for i in range(23,len(lines)):
        fid.write("%s" %(lines[i]))
    
    fid.close()
    
def write_par_SGT(Si):
    
    sname='e3d-fx.par'
    os.system('cp SGT/e3d-fx_default.par e3d-fx.par')
    fid=open(sname,'a')
    fid.write("%s\n" %('xsrc='+str(int(Si[0]))))
    fid.write("%s\n" %('ysrc='+str(int(Si[1]))))
    fid.write("%s\n" %('zsrc=1'))    
    fid.close()
    
    sname='e3d-fy.par'
    os.system('cp SGT/e3d-fy_default.par e3d-fy.par')
    fid=open(sname,'a')
    fid.write("%s\n" %('xsrc='+str(int(Si[0]))))
    fid.write("%s\n" %('ysrc='+str(int(Si[1]))))
    fid.write("%s\n" %('zsrc=1'))    
    fid.close()

    sname='e3d-fz.par'
    os.system('cp SGT/e3d-fz_default.par e3d-fz.par')
    fid=open(sname,'a')
    fid.write("%s\n" %('xsrc='+str(int(Si[0]))))
    fid.write("%s\n" %('ysrc='+str(int(Si[1]))))
    fid.write("%s\n" %('zsrc=1'))    
    fid.close() 
    
#def write_cord_SGT(S_all):    
#    AZ_TRO-h0.100.cordfile
#    sname='SGT/AAA.cordfile'
#    os.system('cp SGT/AAA_default.cordfile AAA.cordfile')    
    
def read_srf_source(source_file):

    with open(source_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nShot=int(line0[0])
    S=np.zeros((nShot,3))
    sNames = []
    for i in range(1,nShot+1):
        line_i=lines[i].split()
        S[i-1,0]=line_i[0]
        S[i-1,1]=line_i[1]
        S[i-1,2]=line_i[2]
        sNames.append(line_i[3])

    return nShot, S, sNames

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



def Vsp_read(nx,ny,nz,snap_file):
    fid=open(snap_file,'r')
    sek=np.fromfile(fid, dtype='<f4')
    Vp=np.reshape(sek,[ny,nz,nx])
    return Vp

# Read_Dev_strain_ex2.m
dt=0.08

def read_stat_name_utm_new(station_file):

    with open(station_file, 'r') as f:
        lines = f.readlines()
    line0=lines[0].split()
    nRec=int(line0[0])
    R=np.zeros((nRec,4))
    statnames = [] 
    for i in range(1,nRec+1):
        line_i=lines[i].split()
        R[i-1,0]=float(line_i[0])
        R[i-1,1]=float(line_i[1])
        statnames.append(line_i[2])
        R[i-1,2]=float(line_i[3])
        R[i-1,3]=float(line_i[4])
    return nRec, R, statnames
############################################################
os.system('cp ../../../../Model/Models/vs3dfile_inv_m10_4_2km.s Model/vs3dfile_opt.s')
os.system('cp ../../../../Model/Models/vp3dfile_inv_m10_4_2km.p Model/vp3dfile_opt.p')
os.system('cp ../../../../Model/Models/rho3dfile_inv_m10_4_2km.d Model/rho3dfile_opt.d')

station_file = '../../../../StatInfo/STATION_utm_dh_2km.txt'  
nRec, R, statnames = read_stat_name_utm_new(station_file)

S = R
#fi1=open('ipart.dat','r')
#ipart=np.int64(np.fromfile(fi1,dtype='int64'))
#fi1.close()    
#ipart=int(ipart)

#os.system('rm -r ../../../AdjSims/V3.0.7-a2a_xyz/Adj-InputAscii/*')

R_all=np.ones([nRec,3,nRec])

statnames_RGT = statnames
#statnames_RGT = ['DSZ','THZ','KHZ','INZ','LTZ','GVZ','OXZ']
#statnames_RGT = ['NNZ','QRZ','WEL']
#statnames_RGT = ['NNZ']

for i,statname in enumerate(statnames):
    if (statname in statnames_RGT):
        os.system('rm OutBin/*')
        ishot=i
        #mkdir_p('../../Vel_es_AB/Vel_es_'+str(ishot))    
        mkdir_p('../../Dev_Strain/'+statname)
        if (2>1):
            fi=open('iShot.dat','w')
            (np.int64(ishot)).tofile(fi)
            fi.close() 
            print('alpha_receiver='+statname)  
            os.system('rm ../../Dev_Strain/*.*')

            print(R[ishot,:])
            #input('-->') 
#            write_par_unit_bd_force_source_i(S[ishot-1,:],0)
            write_par_SGT(R[ishot,0:2])
            #input('-->') 
            write_slurm_SGT(statname,R[ishot,2],R[ishot,3])
            
            job_file11 = 'sgtrun-mpi.sl'       
            job_submitted(job_file11)
            #nom-mpi merging 3 sgt-s
            os.system('./merge_sgt-nonmpi.csh')    
            print("fw-xyz finished")
            mkdir_p('../../Dev_Strain/'+statname)
            os.system('mv SgtFiles/*.* ../../Dev_Strain/'+statname)



    
