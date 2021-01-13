#!/usr/bin/env python

import os
import numpy as np
import obspy
#
from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens, plot_beachball, plot_misfit_dc
from mtuq.grid import DoubleCoupleGridRegular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
#from mtuq.process_data import ProcessData
from mtuq.process_data_NZ import ProcessData
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.io.readers.SAC_NZ import read as readsac

from obspy import UTCDateTime
import obspy.signal.rotate as rotate
from scipy.signal import resample
import matplotlib.pyplot as plt
from qcore import timeseries

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

#Number of time samples, time step and time vector defined similar to the synthetic data from emod3d:
nt=2500
dt=0.08
GV=['090','000','ver']
statnames_BB = ['QRZ','NNZ','WEL','DSZ','THZ','KHZ','INZ','LTZ','GVZ','OXZ'] 
scale_factor_HH = 1.0e-9 #pysep/util_write_cap.py

if __name__=='__main__':
    #
    # Carries out grid search over 64,000 double-couple moment tensors
    #
    # USAGE
    #   python SerialGridSearch.DoubleCouple.py
    #
    # A typical runtime is about 60 seconds. For faster results try 
    # GridSearch.DoubleCouple.py, which runs the same inversion in parallel
    #


    #
    # We will investigate the source process of an Mw~4 earthquake using data
    # from a regional seismic array
    #

    #path_data=    fullpath('data/examples/20090407201255351/*.[zrt]')
#    path_data=    fullpath('C:/UC/mtuq/examples/*.[SAC]')
#    path_weights= fullpath('data/examples/20090407201255351/weights.dat')
#    event_id=     '20090407201255351'
#    model=        'ak135'

    path_data=    fullpath('data/examples/20130721151511707/*.[zrt]')
    path_weights= fullpath('data/examples/20130721151511707/weight.dat')
    event_id=     '20130721151511707'
    model=        'ak135'

    #
    # Body and surface wave measurements will be made separately
    #

#    process_bw = ProcessData(
#        filter_type='Bandpass',
#        freq_min= 0.1,
#        freq_max= 0.333,
#        pick_type='taup',
#        taup_model=model,
#        window_type='body_wave',
#        window_length=15.,
#        capuaf_file=path_weights,
#        )
#
#    process_sw = ProcessData(
#        filter_type='Bandpass',
#        freq_min=0.025,
#        freq_max=0.0625,
#        pick_type='taup',
#        taup_model=model,
#        window_type='surface_wave',
#        window_length=150.,
#        capuaf_file=path_weights,
#        )

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
    #
    # For our objective function, we will use a sum of body and surface wave
    # contributions
    #

    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T'],
        )


    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)


    #
    # Next, we specify the moment tensor grid and source-time function
    #

    grid = DoubleCoupleGridRegular(
        npts_per_axis=40,
        magnitudes=[4.5])

    wavelet = Trapezoid(
        magnitude=4.5)


    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #

#    origin = Origin({
#        'time': '2009-04-07T20:12:55.000000Z',
#        'latitude': 61.454200744628906,
#        'longitude': -149.7427978515625,
#        'depth_in_m': 33033.599853515625,
#        'id': '20090407201255351'
#        })

    origin = Origin({
        'time': '2013-07-21T15:15:11.707Z',
        'latitude': -41.450298,
        'longitude': -41.450298,
        'depth_in_m': 10000.0,
        'id': '2013p544960'
        })
    #
    # The main I/O work starts now
    #

    print('Reading data...\n')
    data = read(path_data, format='sac',
        event_id=event_id,
        station_id_list=station_id_list,
#        tags=['units:cm', 'type:velocity']) 
        tags=['units:m', 'type:velocity'])         
    
#    data = readsac(path_data, format='sac',
#        event_id=event_id,
#        station_id_list=station_id_list,
#        tags=['units:cm', 'type:velocity'])

    data.sort_by_distance()
    stations = data.get_stations()


    print('Processing data...\n')
    data_bw = data.map(process_bw)
    data_sw = data.map(process_sw)
    
    mkdir_p('SAVE_data_bw')
    mkdir_p('SAVE_data_sw')
    mainfolder_bw = 'SAVE_data_bw/'
    mainfolder_sw = 'SAVE_data_sw/'    
    
    os.system('rm SAVE_data_bw/*')
    os.system('rm SAVE_data_sw/*')    
    for i1 in range(0,len(data_bw)):
#    for i1 in range(0,1):  

        if (data[i1][0].id[3:6] in statnames_BB):
            data_bw[i1][0].stats.sac['dist'], data_bw[i1][0].stats.sac['az'], data_bw[i1][0].stats.sac['baz'] = \
                obspy.geodetics.gps2dist_azimuth(
                    data_bw[i1][0].stats.sac['evla'], data_bw[i1][0].stats.sac['evlo'],
                    data_bw[i1][0].stats.sac['stla'], data_bw[i1][0].stats.sac['stlo'])  
                
            data_bw[i1][0].stats.back_azimuth = data_bw[i1][0].stats.sac['baz']
            data_sw[i1][0].stats.back_azimuth = data_bw[i1][0].stats.sac['baz']
            #Rotate back to NEZ    
            data_bw[i1].rotate('RT->NE')    
            data_sw[i1].rotate('RT->NE')
             
            for i2 in range(0,3):   
            
                statname = data_bw[i1][i2].id
                s0 = origin.id + "." + statname[0:-7]
    #            starttime_wd = UTCDateTime(str(data[i1][i2]).split(' |')[1].split(' - ')[0])
    #            starttime_ev = UTCDateTime(origin['time'])     
    #            start0 = int((starttime_wd-starttime_ev)/0.02)
    #            plt.plot(data[i1][i2])
    #            plt.show()            
                    
                starttime_wd = UTCDateTime(str(data_bw[i1][i2]).split(' |')[1].split(' - ')[0])
                starttime_ev = UTCDateTime(origin['time'])
                data_seg = resample(data_bw[i1][i2].data,int(len(data_bw[i1][i2].data)*0.02/dt))*scale_factor_HH
                data_full = np.zeros(nt)
                nt_start = int((starttime_wd-starttime_ev)/dt)
                data_full[nt_start:nt_start+len(data_seg)] = data_seg
                timeseries.seis2txt(data_full,dt,mainfolder_bw,s0,statname[-3:])
                plt.plot(data_full)
    #            plt.ylim([-2*10**6,2*10**6])
                plt.ylim([-2*10**(-3),2*10**(-3)])     
                plt.show()
                
                starttime_wd = UTCDateTime(str(data_sw[i1][i2]).split(' |')[1].split(' - ')[0])
    #            starttime_ev = UTCDateTime(origin['time'])
                data_seg = resample(data_sw[i1][i2].data,int(len(data_sw[i1][i2].data)*0.02/dt))*scale_factor_HH
                data_full = np.zeros(nt)
                nt_start = int((starttime_wd-starttime_ev)/dt)
                data_full[nt_start:nt_start+len(data_seg)] = data_seg
                timeseries.seis2txt(data_full,dt,mainfolder_sw,s0,statname[-3:])            
                plt.plot(data_full)
#                plt.ylim([-2*10**6,2*10**6])        
                plt.ylim([-2*10**(-3),2*10**(-3)])                    
                plt.show()
        else:
            print('station not in the list!')
             

            
        



