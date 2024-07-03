import numpy as np
from functools import lru_cache
from random import shuffle
from collections import defaultdict

# from liouvillian import Interaction, Onsites, Dissipation, Lindblad, Lindblad_new
from liouvillian import Lindblad

import utility as ut
import stats as sts
from liouville_space import liouville
from driver import steady_state
from sampler import exact, mcmc
from vstate import vqs
from optimizer import NatGradExact, NatGradSampling
from machine import rbm
from observable import local_observable
from lattice import Linear_Lattice, Triangular_Lattice

import machine_cy as mcy

import os
import pickle

# directory = r'//home/simon/Desktop/PhD/Projects/RBM/liouville_net/Data/ReportData/xx_model/aa_LDM/new'

###############################################################################################################
sx = np.array([[0,1],[1,0]])
sz = np.array([[1,0],[0,-1]])
id = np.eye(2)
sm = np.array([[0,0],[1,0]])
sp = np.array([[0,1],[0,0]])

###############################################################################################################
ni = 2000
ns = 300
nc = 4
nsweeps = 20
nd = 500

beta = 1
eta = 0.01
reg = 0.01

L = 6


J = 0
h = 5
g = 1

'''
beta = 1, ns = 4500, eta = 0.001, reg = 0.001
beta = 2, ns = 9000, eta = 0.001, reg = 0.01
beta = 2, ns = 18000, eta = 0.001, reg = 0.01
'''

###############################################################################################################
la = Linear_Lattice(L,ndim=1,pbc=False)
N = la.nsites

li = liouville(la)

ln = Lindblad(lattice=la)
# ln.add_term(op1 = (J/4)*sx, op2 = sx)
for i in range(N):
    ln.add_dissipation(op1 = g * sm, s = i)
    ln.add_term(op1 = (h/2)*sx, s=i)

ln.build_gates()

ma = rbm(visible=N,beta=beta)#h,v
sa = mcmc(li,nchains = nc ,nsweeps = nsweeps)

vs = vqs(sa,ma,nsamples=ns,ndiag=nd)#
# vs.set_random(O=1e-4,seed=123)
vs.set_random(O=1e-4,seed=123)
vs.sampling()

op = NatGradSampling(vs,ln,eta=eta,l=reg) ## xx model

ss = steady_state(op)

# try:
ss.run(niter=ni,Verbose=True,log_full_state = True)
# except:
#     vs.set_random(O=1e-4,seed=123)
#     ss.run(niter=ni,Verbose=True,p_log=250)


# observables
zN =  ss.expectation(local_observable(N,np.array([sz]),np.array([0])))
xN =  ss.expectation(local_observable(N,np.array([sx]),np.array([0])))
#
zzN =  ss.expectation(local_observable(N,np.array([sz,sz]),np.array([0,1])))
xxN =  ss.expectation(local_observable(N,np.array([sx,sx]),np.array([0,1])))

# a = np.linspace(0,4,9)
# idx = np.where(np.abs(a-h)<1e-14)[0][0]
#
di = {"L": ss.stats_history[ss.epoch],
   "exp":{"x": xN,
          "z": zN
  },
   "cor":{"xx": xxN,
               "zz": zzN
         }
  }


# fname = f"LDM_xx_N{N}_beta{beta}_eta{eta}_reg{reg}_ns_{ns*4}_{idx}.p"
pickle.dump(ss.stats_history, open( "h=5_o=1e-4.p", "wb" ))

'''
END
'''



# TakeTime = True
# #
# def main():
#     # ss = steady_state(op)
#     ss.run(niter=ni,Verbose=True)
# if __name__ == "__main__":
#
# 	if(TakeTime == True):
# 		import cProfile
# 		cProfile.run('main()', "profile.dat")
#
# 		import pstats
# 		from pstats import SortKey
#
# 		with open("output_time_cy.txt", "w") as f:
# 			p = pstats.Stats("profile.dat", stream=f)
# 			p.sort_stats("cumtime").print_stats()
#
#
# 		with open("output_calls_cy.txt", "w") as f:
# 			p = pstats.Stats("profile.dat", stream=f)
# 			p.sort_stats("calls").print_stats()
