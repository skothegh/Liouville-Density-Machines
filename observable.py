import numpy as np
import utility as ut
from collections import defaultdict

def idx_new(Nv, config):
    '''
    Nv: number of sites
    config: list of local states [0,1,2,3,2,...]

    returns: index of state
    '''
    idx = 0
    for i in range(Nv):
        idx += config[i] * 4**(Nv - 1 - i)
    return int(idx)

def config_new(Nv,i):
    '''
    Nv: number of sites (visible unites)
    i: index of a state

    return: configuration
    '''
    d = 4 ** Nv
    ds = d // 4

    config = np.zeros(Nv,dtype=np.int32)

    for j in range(Nv):
        config[j] = (i % (d / 4**(j) )) //  (ds // 4**j)

    return config


class local_observable():
    '''
    generates a local observable which can be efficiently estimated using markov samples
    nsites: number of sites in chain?
    terms: list of two spin-1/2 operators
    sites: list of sites the operators act on
    '''

    def __init__(self,nsites,terms,sites):
        # assert len(terms) == 1 # we only look at two site correlators

        assert terms.shape[0] < 3, "at most 2-site correlators"
        if(terms.shape[0] == 2):
            assert sites.shape[0] == 2
            assert sites[0] < sites[1]

        # assert nsites < sites.max(), "can't measure outside the chain..."

        self.nsites = nsites # length of chain
        self.terms = terms
        self.sites = sites # operators act on
        self.nterms = terms.shape[0] # number of terms

        self.bond = None
        self.gate = None

        self.set_gate()

    def set_gate(self):
        '''
        creates a dense matrix which contains all relevant terms for the expectation value
        '''
        if(self.nterms == 1): #single-site expectation
            assert len(self.sites) == 1

            if(self.sites[0] == self.nsites-1): #last site
                self.gate = np.kron(np.eye(4),ut.left_op(self.terms[0]))
                self.bond = self.sites[0] - 1

            else: # any other site
                self.gate = np.kron(ut.left_op(self.terms[0]),np.eye(4))
                self.bond = self.sites[0]

        elif(np.abs(self.sites[0] - self.sites[1]) == 1): #nearest neighbors correlation
            assert len(self.sites) == 2

            # self.gate = ut.left_op(np.kron(self.terms[0],self.terms[1]))
            self.gate = np.kron(ut.left_op(self.terms[0]),ut.left_op(self.terms[1]))

        else:
            assert len(self.sites) == 2

            self.gate = np.kron(ut.left_op(self.terms[0]),ut.left_op(self.terms[1]))

    def get_row(self,config):
        if(len(self.sites) == 1):
            return self._get_row_onsite(config)

        else:
            assert len(self.sites) == 2

            return self._get_row_inter(config)

    def _get_row_onsite(self,config):
        d = defaultdict(lambda : 0)

        # for i in range(gate.shape[0]):
        i = self.bond
        ct = np.copy(config)
        row = self.gate[idx_new(2,config[i:2+i])]#at most nearest neighbour
        for j in range(16):
            if(not row[j] == 0):
                ct[i:2+i] = config_new(2,j)
                d[tuple(ct)] += row[j]

        return d

    def _get_row_inter(self,config):
        d = defaultdict(lambda : 0)

        # for i in range(gate.shape[0]):
        ct = np.copy(config)

        c0 = config[self.sites[0]]
        c1 = config[self.sites[1]]

        c = np.array([c0,c1])


        row = self.gate[idx_new(2,c)] #at most nearest neighbour
        for j in range(16):
            if(not row[j] == 0):
                temp = config_new(2,j)
                ct[self.sites[0]] = temp[0]
                ct[self.sites[1]] = temp[1]
                d[tuple(ct)] += row[j]

        return d
