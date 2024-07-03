import numpy as np
from functools import lru_cache
from scipy.linalg import sqrtm

import machine_cy as mcy

###### DISTANCE MEASURES
def Fidelity(a,b):
    '''
    density matrix fidelity. requires hermitian, positive, trace = 1 for a and b
    '''
    return ((np.trace( sqrtm( sqrtm(a) @ b @ sqrtm(a) ) ) )**2)

def tn(rho):
    '''
    trace norm
    '''
    return np.trace(sqrtm(rho.conj().T@rho))

def dag(a):
    return a.conj().T

def distance(a,b):
    return np.sqrt(np.trace((a-b)@hc(a-b)))

def distance2(a,b):
    return np.sum(np.abs(a) - np.abs(b))**2

def absolute_relative_distance(P,Q):
    '''
    NOT A METRIC!
    Q has to be the exact distribution
    '''
    assert P.shape[0] == Q.shape[0]

    X = P.shape[0]
    d = 0
    for x in range(X):
        if(Q[x] == 0):
            if(Q[x] == P[x]):
                d += 0
            else:
                d += np.inf
        else:
            d += np.abs(P[x]-Q[x]) / np.abs(Q[x])

    return d

def overlap(v1,v2):
    # return (sqrt(v1)@sqrt(v2))**2
    return np.conj(v1)@v2/np.sqrt(np.conj(v1)@v1 * np.conj(v2)@v2)

def sq_ed(v1,v2):
    return 1 - np.sum(np.abs(v1 - v2))

def KL_Div(P,Q): # (P|Q)
    assert P.shape[0] == Q.shape[0]

    X = P.shape[0]
    d = 0
    for x in range(X):
        if(P[x] == 0):
            d += 0
        elif(Q[x] == 0):
            d += np.nan
        else:
            d += P[x] * np.log(P[x]/Q[x])
    return d


def get_idx(Nv, config):
    idx = 0
    for i in range(Nv):
        idx += config[i] * 4**(Nv - 1 - i)
    return int(idx)

def get_config(Nv,i):
    d = 4 ** Nv
    ds = d // 4

    config = np.zeros(Nv,dtype=np.int64)

    for j in range(Nv):
        config[j] = (i % (d / 4**(j) )) //  (ds // 4**j)

    return config

def get_binary(Nv,i):
    d = 2 ** Nv
    ds = d // 2

    config = np.zeros(Nv,dtype=np.int64)

    for j in range(Nv):
        config[j] = (i % (d / 2**(j) )) //  (ds // 2**j)

    return config

def split_con(config):
    sc = np.array([])
    N = len(config)
    for i in range(N):
        sc = np.concatenate((sc,get_binary(2,config[i])),axis=None)
    return sc


def get_qt_idx(config):
    N = len(config)
    Nsc = 2*N

    Ns = 4**N
    sc = split_con(config)

    idx = 0
    for i in range(Nsc):
        idx += sc[i] * 2**(N-1) * np.sqrt(Ns)**(i%2) / (2**(i//2))
    return int(idx)




def hc(O):
    return conj(O).T

def kp(a,b):
    l = len(a)*len(b)
    a = a[:,newaxis]
    b = b[newaxis,:]
    return reshape(a*b,l)

def bins(N,samples):
    b = np.zeros(4**N)
    for i in samples:
        b[mcy.idx_cy(N,i)] += 1
    return b


@lru_cache(maxsize=None)
def config_cached(Nv,i):
    # relabel + new basis
    return mcy.config_cy(Nv,i)

def relabel(c):
    temp = np.copy(c)
    for i in range(len(c)):
        temp[i] = label_cache(c[i])
    return temp

@lru_cache(maxsize=4)
def label_cache(i):
    if(i == 0):
        s = 2
    elif(i == 1):
        s = 1
    elif(i == 2):
        s = -1
    elif(i == 3):
        s = -2
    return s

def left_op(operator):
    '''
    calculates a operator version in liouville space of an operator that acts from the left in hilbert space,
    i.e.
    H * rho -> H_left * |rho>>
    '''
	size = operator.shape
	Id = np.eye(size[0])
	left = np.kron(operator,Id)
	return left

def right_op(operator):	# builds liouville op of left acting hilbert op: H*rho -> H_l |rho>>
    '''
    calculates a operator version in liouville space of an operator that acts from the right in hilbert space,
    i.e.
    rho * H -> H_right * |rho>>    
    '''
	size = operator.shape
	Id = np.identity(size[0])
	right = np.kron(Id,operator.T)
	return right
