
import obspy
import numpy as np

from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase



class GreensTensor(GreensTensorBase):
    """
    SPECFEM3D Green's tensor object

    .. note:

      SPECFEM3D Green's functions describe the impulse response of a general
      inhomogeneous medium.  Time series represent vertical, radial, and 
      transverse displacement in units of m*(N-m)^-1 

      (FIXME: DOUBLE CHECK UNITS)

    """
    def __init__(self, *args, **kwargs):
        super(GreensTensor, self).__init__(*args, **kwargs)

        import warnings
        warnings.warn("SPECFEM3D modules have not yet been tested.")

        if 'type:greens' not in self.tags:
            self.tags += ['type:greens']

        if 'type:displacement' not in self.tags:
            self.tags += ['type:displacement']

        if 'units:m' not in self.tags:
            self.tags += ['units:m']




    def _precompute(self):
        """ Computes numpy arrays used by get_synthetics
        """
        if self.include_mt:
            self._precompute_mt()

        if self.include_force:
            self._precompute_force()

        self._permute_array()


    def _precompute_mt(self):
        """ Assembles NumPy arrays used in moment tensor linear combination
        """

        array = self._array
        phi = np.deg2rad(self.azimuth)
        _j = 0

        for _i, component in enumerate(self.components):
            if component=='Z':
#                array[_i, _j+0, :] = self.select(channel="Z.Mrr")[0].data
#                array[_i, _j+1, :] = self.select(channel="Z.Mtt")[0].data
#                array[_i, _j+2, :] = self.select(channel="Z.Mpp")[0].data
#                array[_i, _j+3, :] = self.select(channel="Z.Mrt")[0].data
#                array[_i, _j+4, :] = self.select(channel="Z.Mrp")[0].data
#                array[_i, _j+5, :] = self.select(channel="Z.Mtp")[0].data
                
                array[_i, _j+0, :] = self.select(channel="Z")[0].data
                array[_i, _j+1, :] = self.select(channel="Z")[1].data
                array[_i, _j+2, :] = self.select(channel="Z")[2].data
                array[_i, _j+3, :] = self.select(channel="Z")[3].data
                array[_i, _j+4, :] = self.select(channel="Z")[4].data
                array[_i, _j+5, :] = self.select(channel="Z")[5].data                

            elif component=='R':
                array[_i, _j+0, :] = self.select(channel="R")[0].data
                array[_i, _j+1, :] = self.select(channel="R")[1].data
                array[_i, _j+2, :] = self.select(channel="R")[2].data
                array[_i, _j+3, :] = self.select(channel="R")[3].data
                array[_i, _j+4, :] = self.select(channel="R")[4].data
                array[_i, _j+5, :] = self.select(channel="R")[5].data

            elif component=='T':
                array[_i, _j+0, :] = self.select(channel="T")[0].data
                array[_i, _j+1, :] = self.select(channel="T")[1].data
                array[_i, _j+2, :] = self.select(channel="T")[2].data
                array[_i, _j+3, :] = self.select(channel="T")[3].data
                array[_i, _j+4, :] = self.select(channel="T")[4].data
                array[_i, _j+5, :] = self.select(channel="T")[5].data


    def _precompute_force(self):
        raise NotImplementedError



    def _permute_array(self):
        # FIXME: DOUBLE CHECK THAT SPECFEM3D WORKS IN UP-SOUTH-EAST
        # CONVENTION
        pass
