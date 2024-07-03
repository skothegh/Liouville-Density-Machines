import numpy as np
import machine_cy as mcy
from functools import lru_cache

from stats import mc_stats

from sampler import exact, mcmc
from utility import relabel



class vqs(): #variational quantum state
    '''
    class storing the variational quantum state, i.e. the machine, the parameters and the samples
    '''
    def __init__(self, sampler, rbm, nsamples=900,ndiag = 200):
        '''
        sa: sampler object
        ma: machine object

        Nv: number of sites
        Nh: number of hidden units

        Np: number of parameters

        samples: [config, rho(config)]
        diag_samples : [diag_configs, rho(diag_configs)]

        '''
        # self.li = liouville
        self.sa = sampler
        self.ma = rbm

        self.Nv = self.ma.Nv
        self.Nh = self.ma.Nh

        self.Np = self.ma.Np

        self.samples = [None,None] #samples, states
        self.diag_samples = [None,None] #samples, states

        self.ns0 = nsamples
        self.nd0 = ndiag
        self.ns = nsamples
        self.nd = ndiag

    def set_random(self,O=0.01,seed=None):
        self.ma.set_rand(O,seed)

    def set_parameters(self,p):
        '''
        set parameters
        '''
        if(not len(p) == self.ma.Np):
            raise Exception(f"{self.ma.Np} parameters are required. {len(p)} parameters passed.")
        self.ma.set_param(p)

    def update(self,updates):
        '''
        update machine parameters
        '''
        self.ma.update(updates)

    def diag_sampling(self):
        '''
        sample from diagonal of density matrix
        '''
        self.diag_samples = self.sa.diag_sample(self.ma.a1,self.ma.a2,self.ma.a3,self.ma.b,self.ma.u,self.ma.v,self.ma.w,self.nd0)

    def sampling(self):
        '''
        sample whole density matrix
        '''
        if(type(self.sa) is exact):
            self.samples = self.sa.sample(self.ma.rho)
            self.ns = len(self.samples[0])
        else:
            self.samples = self.sa.sample(self.ma.a1,self.ma.a2,self.ma.a3,self.ma.b,self.ma.u,self.ma.v,self.ma.w,self.ns0)
            self.ns = len(self.samples[0][0])

    def vs_state(self):
        '''
        return a density matrix. currently only supports Nv=6
        '''
        if(type(self.sa) is exact):
            return self._vs_state_exact()
        else:
            return self._vs_state_sample()

    def _vs_state_exact(self):
        k = np.zeros((4 ** self.Nv),dtype=np.complex128)

        for i in range(4 ** self.Nv):
            k[i] = self.samples[1][i]

        if self.Nv == 2:
            v = k.reshape(2,2,2,2)
            v = np.einsum("abcd->acbd",v).reshape(4,4)
            v /= v.trace()

        return v

    def _vs_state_sample(self):
        nchains = len(self.samples[0])
        nsamples, nsites = self.samples[0][0].shape

        k = np.zeros((4 ** self.Nv),dtype=np.complex128)

        for h in range(nchains):
            for i in range(nsamples):
                k[self.sa.li.idx(self.samples[0][h][i])] += self.samples[1][h][i]

        d = int(np.sqrt(4 ** self.Nv) + 1)

        if nsites == 1:
            v = k.reshape(2,2)
            v /= v.trace()

        if nsites == 2:
            v = k.reshape(2,2,2,2)
            v = np.einsum("abcd->acbd",v).reshape(4,4)
            v /= v.trace()

        if nsites == 3:
            v = k.reshape(2,2,2,2,2,2)
            v = np.einsum("abcdef->acebdf",v).reshape(8,8)
            v /= v.trace()

        if nsites == 4:
            v = k.reshape(2,2,2,2,2,2,2,2)
            v = np.einsum("abcdefgh->acegbdfh",v).reshape(16,16)
            v /= v.trace()

        if nsites == 5:
            v = k.reshape(2,2,2,2,2,2,2,2,2,2)
            v = np.einsum("abcdefghij->acegibdfhj",v).reshape(32,32)
            v /= v.trace()

        if nsites == 6:
            v = k.reshape(2,2,2,2,2,2,2,2,2,2,2,2)
            v = np.einsum("abcdefghijkl->acegikbdfhjl",v).reshape(64,64)
            v /= v.trace()

        return v

    def reset(self):
        '''
        resets the state
        '''
        self.samples = [None,None]
        self.diag_samples = [None,None]
        # self.ns = None
        self.idx_rho.cache_clear()

    def probability(self):
        '''
        calculates probability distriburion given the samples
        '''
        # if(self.samples[1] is None): self.sampling()

        d = np.zeros(4**self.Nv,dtype=np.complex128)
        for i in range(len(self.samples[0])):
            d[self.sa.li.idx(self.samples[0][i])] += self.samples[1][i]
        d = np.conj(d)*d/(np.conj(d)@d)
        return d

    def expectation(self,observable):
        '''
        takes a local_observable object and estimates its mean and variance given diagonal samples
        '''
        self.diag_sampling()
        l = self._local_observable(observable)
        return mc_stats(l)

    def _local_observable(self,observable):
        nchains, nsamples, nvisible = self.diag_samples[0].shape

        loc_ob = np.zeros((nchains,nsamples),dtype=np.complex128)

        for h in range(nchains):
            for i in range(nsamples):
                d = observable.get_row(self.diag_samples[0][h][i])
                for key in d:
                    loc_ob[h,i] += d[key] * self.rho(np.array(key))  / self.diag_samples[1][h][i]

        return loc_ob

    def rho(self,c):
        '''
        calls the state function of the machine
        '''
        return self.ma.rho(c)

    @lru_cache(maxsize=None)
    def idx_rho(self,j):
        '''
        calculate the density function given the index of configuration. haches the index to
        minimize calls to config(index) function
        '''
        return self.ma.rho(self.sa.li.config_wrap(j))


#END



#deprecated
'''
    def to_ket(self):
        if(type(self.sa) is exact):
            return self._to_ket_exact()
        else:
            return self._to_ket_sample()


    def _to_ket_exact(self):
        k = np.zeros(4 ** self.Nv)

        t0 = np.array([1,0,0,1])
        t = np.copy(t0)
        for i in range(1,self.Nv):
            t = np.kron(t,t0)

        for i in range(4 ** self.Nv):
            k[i] = self.samples[1][i]

        kn = k / (t@k)

        return kn

    def _to_ket_sample(self):
        nchains = len(self.samples[0])
        nsamples, nsites = self.samples[0][0].shape


        t0 = np.array([1,0,0,1])
        t = np.copy(t0)
        for i in range(1,self.Nv):
            t = np.kron(t,t0)

        k = np.zeros((4 ** self.Nv,2),dtype=np.complex128)

        for h in range(nchains):
            for i in range(nsamples):
                k[self.sa.li.idx(self.samples[0][h][i])] [0] += self.samples[1][h][i]
                k[self.sa.li.idx(self.samples[0][h][i])] [1] += 1

        kn = k.T[0] / (k.T[1] + 1)

        # k = k / self.ns
        d = int(np.sqrt(4 ** self.Nv) + 1)
        kn = kn / (kn@t)

        return kn

    def to_matrix(self):
        v = self._to_ket_sample()
        v = v.reshape(2,2,2,2)
        v = np.einsum("abcd->acbd",v).reshape(4,4)
        return v
'''
