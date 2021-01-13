
import obspy
import numpy as np

from obspy.core import Stream
from mtuq.greens_tensor.syngine import GreensTensor 
from mtuq.io.clients.base import Client as ClientBase
from mtuq.util.signal import resample
from obspy import UTCDateTime

SUFFIXES = [
    'Z.Mrr',
    'Z.Mtt',
    'Z.Mpp',
    'Z.Mrt',
    'Z.Mrp',
    'Z.Mtp',
    'R.Mrr',
    'R.Mtt',
    'R.Mpp',
    'R.Mrt',
    'R.Mrp',
    'R.Mtp',
    'T.Mrr',
    'T.Mtt',
    'T.Mpp',
    'T.Mrt',
    'T.Mrp',
    'T.Mtp',
    ]


class Client(ClientBase):
    """ SPECFEM3D Green's tensor client

    .. rubric:: Usage

    To instantiate a database client, supply a path or url:

    .. code::

        from mtuq.io.clients.SPECFEM3D_SAC import Client
        db = Client(path_or_url)

    Then the database client can be used to generate GreensTensors:

    .. code::

        greens_tensors = db.get_greens_tensors(stations, origin)


    .. note::

    """

    def __init__(self, path_or_url=None, model=None, 
                 include_mt=True, include_force=False):

        self.path = path_or_url

        self.include_mt = include_mt
        self.include_force = include_force


    def get_greens_tensors(self, stations=[], origins=[], verbose=False):
        """ Reads Green's tensors

        Returns a ``GreensTensorList`` in which each element corresponds to a
        (station, origin) pair from the given lists

        .. rubric :: Input arguments

        ``stations`` (`list` of `mtuq.Station` objects)

        ``origins`` (`list` of `mtuq.Origin` objects)

        ``verbose`` (`bool`)

        """
        return super(Client, self).get_greens_tensors(stations, origins, verbose)


    def _get_greens_tensor(self, station=None, origin=None):
        stream = Stream()

        # read data
        stream = Stream()
        stream.id = station.id

        if self.include_mt:
            prefix = station.id
            for suffix in SUFFIXES:
                trace = obspy.read(
                    self.path+'/'+prefix+'.'+suffix+'.sac', format='sac')[0]
                trace.component = suffix
                stream += trace

        if self.include_force:
            raise NotImplementedError


        # what are the start and end times of the data?
#        t1_new = float(station.starttime)
#        t2_new = float(station.endtime)
#        dt_new = float(station.delta)
        t1_new = float(UTCDateTime(2013, 7, 21, 15, 13, 31, 578399))
        t2_new = float(UTCDateTime(2013, 7, 21, 15, 20, 11, 978399))
        dt_new = 0.02        

        # what are the start and end times of the Green's function?
#        t1_old = float(stream[0].stats.starttime)
#        t2_old = float(stream[0].stats.endtime)
#        dt_old = float(stream[0].stats.delta)
        t1_old = float(UTCDateTime('2013-07-21T15:15:11.707Z'))
        t2_old = t1_old+200-0.08
        dt_old = 0.08
        
        for trace in stream:
            # resample Green's functions
            data_old = trace.data
            data_new = resample(data_old, t1_old, t2_old, dt_old,
                                          t1_new, t2_new, dt_new)
            trace.data = data_new
            trace.stats.starttime = t1_new
            trace.stats.delta = dt_new
            trace.stats.npts = len(data_new)

        tags = [
#            'model:%s' % self.model,
            'solver:%s' % 'SPECFEM3D',
             ]

        return GreensTensor(traces=[trace for trace in stream], 
            station=station, origin=origin, tags=tags,
            include_mt=self.include_mt, include_force=self.include_force)

