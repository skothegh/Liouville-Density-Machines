import numpy as np
import machine_cy as mcy

from utility import config_cached

class liouville():
    '''
    Liouville space class. Indexes the configuratins
    '''

    def __init__(self,lattice):
        self.lattice = lattice
        self.Nv = lattice.nsites
        self.dim = 4**lattice.nsites

    def idx(self,config):
        ### Possibly necessary?
        return mcy.idx_cy(self.Nv,config)

    def config_wrap(self,j):
        # relabel + new basis
        return np.array(config_cached(self.Nv,j))

    def c_list(self):
        ns = 4**self.Nv

        con = np.zeros((ns,self.Nv),dtype=np.int64)

        for i in range(ns):
            con[i,:] = self.config_wrap(i)
        return con




#End
