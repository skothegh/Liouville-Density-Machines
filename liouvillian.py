import numpy as np
from numpy import kron, eye, conj
from collections import defaultdict
import utility as ut
# from machine_cy import idx_cy3, config_cy3, get_conn_col_cy
from functools import lru_cache
import machine_cy as mcy


class Lindblad():
    '''
    Lindblad operator class. Takes spin-1/2 operators and turns them into a list of bond gates
    '''
    def __init__(self,lattice):
        self.lattice = lattice

        self.gates = None
        self.Tgates = None
        self.ngates = self.lattice.nbonds

        self.terms = {
                            "inter": [],
                            "field": defaultdict(lambda: np.zeros((2,2)) )
                     }

        self.dissipations = {
                            "inter": [],
                            "field": defaultdict(lambda: np.zeros((2,2)) )
                             }


    def __add__(self,x):
        '''
        adds the gates of two Hamiltonian objects together and returns a new Hamiltonian
        x: Lindbladian
        '''

        assert isinstance(x,Lindblad)
        assert x.lattice == self.lattice

        gates = np.zeros_like(self.gates)

        for b in range(self.ngates):
            assert np.allclose( self.lattice.bonds[b], x.lattice.bonds[b]  )
            gates[b] = self.gates[b] + x.gates[b]

        h = Lindblad(self.lattice)
        h.gates = gates
        h.ngates = self.ngates
        h.terms = {
                   "inter": self.terms["inter"] +  x.terms["inter"],
                   "field": {i: self.terms["field"][i] + x.terms["field"][i]
                                            for i in range(self.lattice.nsites)}
                  }

        return h

    def __mul__(self,x):
        '''
        x: scalar

        multiplies the gates of a Hamiltonian with a scalar
        '''
        assert not hasattr(x,"__len__"), "not a scalar"

        gates = np.zeros_like(self.gates)
        for b in range(self.ngates):
            gates[b] = self.gates[b] * x

        h = Lindblad(self.lattice)
        h.gates = gates
        h.ngates = self.ngates
        h.terms = self.terms

        return h


    def add_term(self,*,op1, op2=None, s = None):
        '''
        only for hamiltonian terms
        '''
        ### asssertions
        if(not s is None):
            assert (0 <= s & s <= self.lattice.nsites - 1), "site out of bounds"

        if(op2 is None):
            assert not s is None, "If we have no interaction, we need a site to act on."


        ### gates
        # interaction
        if(not op2 is None):
            op1_l = ut.left_op(op1)
            op1_r = ut.right_op(op1)

            op2_l = ut.left_op(op2)
            op2_r = ut.right_op(op2)

            g = -1j*(np.kron(op1_l,op2_l) - np.kron(op1_r,op2_r))

            self.terms["inter"].append(g) #append.interaction_gate(op1,op2)

        # field
        elif(not s is None):
            assert op2 is None, "Fields only have an onsite-term."
            self.terms["field"][(s)] += op1


    def add_dissipation(self,*,op1, op2=None, s = None):
        '''
        dissipations
        '''
        ### assertions
        if(not s is None):
            assert (0 <= s & s <= self.lattice.nsites - 1), "site out of bounds"

        if(op2 is None):
            assert not s is None, "If we have no interaction, we need a site to act on."


        ### gates
        # two-site dissipation
        if(not op2 is None):
            D = (
                 kron(kron(op1,np.conj(op1)),kron(op2,np.conj(op2))) -
                 0.5 * (
                        kron(
                             kron(conj(op1.T)@op1, eye(2)),
                             kron(conj(op2.T)@op2, eye(2))
                            ) +
                        kron(
                             kron(eye(2),(conj(op1.T)@op1).T),
                             kron(eye(2),(conj(op2.T)@op2).T)
                            )
                       )
                )
            self.dissipations["inter"].append(D)

        # single-site dissipation
        elif(not s is None):
            assert op2 is None, "onsite-dissipation cannot have a second term"
            self.dissipations["field"][(s)] += op1


    def field_gate(self,op1,op2):
        '''
        takes an operator and returns the 16x16 liouville-space gate


        is called in the build_gates function and draws from terms. reason is:
        the gate depends on the operators on each site! and we only know shich site
        we are looking at in the build_gate function.

        TO TEST: multiple field terms. might be that we cannot simply add them up
        and need to go to liouville first. in which case the structure needs to be reworked..

        '''

        op1_l = ut.left_op(op1)
        op1_r = ut.right_op(op1)

        op2_l = ut.left_op(op2)
        op2_r = ut.right_op(op2)

        sI = -1j*(np.kron(op1_l,eye(4)) - np.kron(op1_r,np.eye(4)))
        Is = -1j*(np.kron(np.eye(4),op2_l) - np.kron(np.eye(4),op2_r))

        return sI, Is

    def dissipation_gate(self,op1,op2):
        D1 = kron(
                   kron(op1,conj(op1)) -
                       0.5 * (kron(eye(2),op1.T@conj(op1)) +
                              kron(conj(op1).T@op1,eye(2))),
                   eye(4)
                  )

        D2 = kron(eye(4),
                  kron(op2,conj(op2)) -
                      0.5 * (kron(eye(2),op2.T@conj(op2)) +
                             kron(conj(op2).T@op2,eye(2)))
                 )

        return D1, D2

    def denominator(self):
        '''
        counts how often a given site appears in bonds. uses that to scale
        field terms on that site.
        '''
        pbc = self.lattice.pbc
        bonds = self.lattice.bonds
        nbonds = self.lattice.nbonds
        nsites = self.lattice.nsites
        dim = self.lattice.ndim

        D = np.zeros((nbonds,2))        # denominator array
        if(pbc):                        # CUBIC LIKE LATTICES! chains, squares, cubes, hypercubes...
            A = np.zeros(nsites)
            for bond in bonds:
                for site in bond:
                    A[site] += 1

        else:
            A = np.zeros(nsites)
            for bond in bonds:
                for site in bond:
                    A[site] += 1

        for b in range(nbonds):
            for s in range(2):
                D[b,s] = 1/A[bonds[b,s]]

        return D


    def build_gates(self):
        nbonds = self.lattice.nbonds

        gates = np.zeros((nbonds,16,16),dtype=np.complex128) #nbonds (pbc), 4x4
        bonds = self.lattice.bonds
        pbc = self.lattice.pbc
        D = self.denominator()

        for b in range(nbonds):
            #### Unitary Gates
            for i in self.terms["inter"]:
                gates[b] += i

            if(len(self.terms["field"]) > 0):
                o1 = self.terms["field"][bonds[b][0]]
                o2 = self.terms["field"][bonds[b][1]]

                sI, Is = self.field_gate(o1,o2)

                gates[b] += (
                             sI * D[b,0] +
                             Is * D[b,1]
                            )

            ### Dissipative Gates
            for d in self.dissipations["inter"]:
                gates[b] += d

            if(len(self.dissipations["field"]) > 0):
                o1 = self.dissipations["field"][bonds[b][0]]   # o-o...-1-2-...-o-o
                o2 = self.dissipations["field"][bonds[b][1]]   # in any dimension

                dI, Id = self.dissipation_gate(o1,o2)

                gates[b] += (
                             dI * D[b,0] +
                             Id * D[b,1]
                            )

        self.gates = gates

    def get_row(self,c):
        return mcy.get_row_new(self.lattice.nsites,self.lattice.bonds,self.gates,c)

    def I(self):
        return np.eye(2)

    def sx(self):
        return np.array([[0,1],[1,0]])

    def sy(self):
        return np.array([[0,-1j],[1j,0]])

    def sz(self):
        return np.array([[1,0],[0,-1]])

    def sm(self):
        return np.array([[0,0],[1,0]])

    def sp(self):
        return np.array([[0,1],[0,0]])

    def n(self):
        return np.array([[1,0],[0,0]])

















#Deprecated
"""
def idx_new(Nv, config):
    idx = 0
    for i in range(Nv):
        # print(i)
        idx += config[i] * 4**(Nv - 1 - i)
    return int(idx)

def config_new(Nv,i):
    d = 4 ** Nv
    ds = d // 4

    config = np.zeros(Nv,dtype=np.int32)

    for j in range(Nv):
        config[j] = (i % (d / 4**(j) )) //  (ds // 4**j)

    return config

def left_op(operator):
	size = operator.shape
	Id = eye(size[0])
	left = kron(operator,Id)
	return left

def right_op(operator):
	size = operator.shape
	Id = eye(size[0])
	left = kron(Id,operator.T)
	return left


class Interaction():
    '''
    Class that structures the interaction terms and strengths.
    Provides a method that returns unitary interaction gates for a specific interaction term and interaction range.
    Provides a method that returns a sum over all interaction terms for a specified interaction length.
    '''

    def __init__(self,terms,J):
        self.iterms = terms
        self.ipar = J

    def ss2(self,ops):
    	op1_l = left_op(ops[0])
    	op1_r = right_op(ops[0])

    	op2_l = left_op(ops[1])
    	op2_r = right_op(ops[1])

    	return kron(op1_l,op2_l) - kron(op1_r,op2_r)

    def igates(self,t):
        '''
        t: number of interaction term
        '''
        g = self.ipar[t]*self.ss2(self.iterms[t])
        return -1j*g

    def tsum(self):
        g = self.ipar[0]*self.ss2(self.iterms[0])

        for t in range(1,len(self.iterms),1):
            g = g + self.ipar[t]*self.ss2(self.iterms[t])
        return -1j*g

class Onsites():
    '''
    Class that structures the interaction terms and strengths.
    Provides a method that returns unitary onsite gates for a specific onsite term.
    Provides a method that returns a sum over all onsite terms for the first and last bond.
    '''

    def __init__(self,terms,h):
        self.oterms = terms
        self.onpar = h

    def on(self,op):
        op_l = left_op(op)
        op_r = right_op(op)

        sI = kron(op_l,eye(4)) - kron(op_r,eye(4))
        Is = kron(eye(4),op_l) - kron(eye(4),op_r)
        return sI, Is

    def ogates(self,o):
        '''
        o: number of onsite term
        '''
        g = self.onpar[o]*self.on(self.oterms[o])[0]/2 + self.onpar[o]*self.on(self.oterms[o])[1]/2
        return -1j*g

    def o_edge(self):
        '''
        o: number of onsite term
        '''
        first = self.onpar[0]*self.on(self.oterms[0])[0]/2
        for o in range(1,len(self.oterms),1):
            first += self.onpar[o]*self.on(self.oterms[o])[0]/2

        last = 	self.onpar[0]*self.on(self.oterms[0])[1]/2
        for o in range(1,len(self.oterms),1):
            last += self.onpar[o]*self.on(self.oterms[o])[1]/2

        return -1j*first, -1j*last

class Dissipation():
    '''
    Class that structures the interaction terms and strengths.
    Provides a method that returns unitary onsite gates for a specific onsite term.
    Provides a method that returns a sum over all onsite terms for the first and last bond.
    '''
    def __init__(self,terms,k):
        self.dterms = terms
        self.dpar = k

    def Dissipator(self,op):
        D1 = kron( kron(op,conj(op)) - 0.5 * (kron(eye(2),op.T@conj(op)) + kron(conj(op).T@op,eye(2))),eye(4))
        D2 = kron(eye(4), kron(op,conj(op)) - 0.5 * (kron(eye(2),op.T@conj(op)) + kron(conj(op).T@op,eye(2))))
        return D1, D2

    def dgates(self,d):
        dis = self.dpar[d]*self.Dissipator(self.dterms[d])[0]/2 + self.dpar[d]*self.Dissipator(self.dterms[d])[1]/2
        return dis

    def d_edge(self):
        first = self.dpar[0]*self.Dissipator(self.dterms[0])[0]/2
        last =  self.dpar[0]*self.Dissipator(self.dterms[0])[1]/2
        return first, last


class Local_Lioville_Operator():
    pass

class Lindblad():
    def __init__(self,N,inter,onsite,diss,pbc = False):
        self.inter = inter
        self.onsite = onsite
        self.diss = diss
        self.gates = None
        self.Tgates = None
        self.ns = N

        self.build_gates()

    def build_gates(self):
        ni = len(self.inter.iterms)
        no = len(self.onsite.oterms)
        nd = len(self.diss.dterms)

        gates = []
        for n in range(self.ns-1): #n bonds

            g = self.inter.igates(0)
            g += self.onsite.ogates(0)
            g += self.diss.dgates(0)

            for t in range(1,ni,1):
                g += self.inter.igates(t)
            for o in range(1,no,1):
                g += self.onsite.ogates(o)
            for c in range(1,nd,1):
                g += self.diss.dgates(c)


            gates.append(g)

        gates[0] += self.onsite.o_edge()[0] + self.diss.d_edge()[0]
        gates[len(gates)-1] += self.onsite.o_edge()[1] + self.diss.d_edge()[1]

        self.gates = np.array(gates,dtype=np.complex128)
        self.Tgates = np.transpose(self.gates,(0,2,1))

        return self.gates

    def get_row(self,config):
        return mcy.get_row(self.gates,config,self.ns)

    def get_col(self,config,conjugate = False):
        return mcy.get_col(self.Tgates,config,self.ns,conjugate)

    def list_col_cons(self,config,conjugate = False):
        return mcy.list_col_cons(self.gates,config,self.ns,conjugate)

    def list_row_cons(self,config):
        '''
        Note! This DOES NOT create the correct list of row connections!
        It creates a list of connections which will be used in LL2 later and
        properly summed over.
        '''
        return mcy.list_row_cons(self.gates,config,self.ns)

    def get_conn_LL(self,config):
        return mcy.get_conn_LL(config,self.ns,self.gates,self.Tgates)

# """
#END
