import numpy as np
import machine_cy as mcy
import multiprocess as mp
from itertools import repeat


class base_sampler():
    '''
    exact and mcmc should iherti from this class
    '''
    pass

class exact(): # "sampler"
    '''
    deosnt' sample but enumerates all basis states of liouville space
    '''
    def __init__(self,liouville):
        self.li = liouville

    def diag_sample(self,a1,a2,a3,b,u,v,w,ndiag,nchains=15):
        return self.refine(mcy.diag_mcmc(a1,a2,a3,b,u,v,w,self.li.Nv,ndiag,nchains))

    def refine(self,raw):
        '''
        removes burn-in and stacks mcmc chains
        '''
        refined = [None,None]

        refined[0] = raw[0][len(raw[0])//10:]
        refined[1] = raw[1][len(raw[0])//10:]

        return refined

    def sample(self,func):
        # confs =
        sts = np.zeros(self.li.dim,dtype=np.complex128)
        con =  self.li.c_list()
        for i in range(self.li.dim):
            sts[i] = func(con[i])
        return [con,sts]

class mcmc():
    '''
    class which contains methods to generate the markov samples
    '''
    def __init__(self,liouville,nchains,nsweeps):
        self.li = liouville
        self.nchains = nchains
        self.nsweeps = nsweeps

    def sample(self,a1,a2,a3,b,u,v,w,ns):
        return self.refine(self.multi_mcmc(a1,a2,a3,b,u,v,w,ns))

    def diag_sample(self,a1,a2,a3,b,u,v,w,ndiag):
        raw = mcy.diag_mcmc(a1,a2,a3,b,u,v,w,self.li.Nv,ndiag,self.nchains)
        return self.sample_sort(raw[0],raw[1])

    def sample_sort(self,configs,states):
        nchains = len(configs)
        nsamples, nsites = configs[0].shape

        for h in range(nchains):
            for i in range(nsites):
                ind = configs[h][:,i].argsort(kind='mergesort')
                configs[h] = configs[h][ind]
                states[h] = states[h][ind]

        return [configs,states]

    def sample_reformat(self,results):
        '''
        might be advantageous to preinitialize arrays instead of appending
        '''
        configs = [results[i][0] for i in range(len(results))]
        states = [results[i][1] for i in range(len(results))]

        # samples = {"Configs":  [results[i][0] for i in range(len(results))],
                     # "Rhos": [results[i][1] for i in range(len(results))]} # for later

        return configs, states

    def refine(self,results):
        refined = self.sample_reformat(results)
        refined = self.sample_sort(refined[0],refined[1])

        return refined

    def multi_mcmc(self,a1,a2,a3,b,u,v,w,Nsamples):
        '''
        a1... u,v,w: parameters of LDM
        Nsamples: number of sampels to be generated
        returns: number of markov chains
        '''
        configs = np.zeros((self.nchains, Nsamples, self.li.Nv),dtype = np.int64)
        states = np.zeros((self.nchains, Nsamples), dtype = np.complex128)


        for i in range(self.nchains):
            configs[i,0], states[i,0] = self.burn_in(a1,a2,a3,b,u,v,w,self.li.Nv,burn_in_period=100)

        np.random.seed(123)
        seeds = np.random.randint(2000,size=self.nchains)
        args = zip(repeat(a1),
                   repeat(a2),
                   repeat(a3),
                   repeat(b),
                   repeat(u),
                   repeat(v),
                   repeat(w),
                   repeat(self.li.Nv),
                   repeat(Nsamples),
                   repeat(self.nsweeps),
                   seeds,
                   configs,
                   states)

        pool = mp.Pool(4)

        results = pool.starmap(mcy.single_chain_cy,args)

        pool.close()

        return results

    def burn_in(self,a1,a2,a3,b,u,v,w,Nsites,burn_in_period):
        return mcy.burn_in(a1,a2,a3,b,u,v,w,Nsites,burn_in_period)

#End
