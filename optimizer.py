import numpy as np
from utility import relabel
import machine_cy as mcy
import stats as sts
from scipy.stats import sem
import multiprocess as mp

import pickle


class Base_Optimizer():
    '''
    Exact and Sampled should inherit from this class!
    '''
    pass

class NatGradExact():
    '''
    optimizes exactly and without sampling. very costly
    '''
    def __init__(self,vqs,lind,eta=0.01,l=1e-4):
        self.eta = eta
        self.reg = l

        self.vqs = vqs #passed each step again
        self.lind = lind #should get initialised as soon as the driver gets init

    def updates(self):
        p = self.probability()

        #### GRADIENT
        L_loc = np.zeros(self.vqs.ns,dtype=np.complex128)
        OE = np.zeros(self.vqs.Np,dtype=np.complex128)
        S1 = np.zeros((self.vqs.Np,self.vqs.Np),dtype = np.complex128)
        S2 = np.zeros((self.vqs.Np,self.vqs.Np),dtype = np.complex128)

        O = np.zeros((self.vqs.ns,self.vqs.Np),dtype=np.complex128)

        for i in range(self.vqs.ns):
            O[i] = self.log_deriv(self.vqs.samples[0][i])
            S1 += np.outer(np.conj(O[i,:]),O[i,:])*p[i]

            lc,lv,k = self.lind.get_row(self.vqs.samples[0][i])

            A = np.zeros((self.vqs.Np),dtype=np.complex128)
            for j in range(k):
                temp = lv[j]*self.vqs.ma.rho(lc[j])/self.vqs.samples[1][i]
                L_loc[i] += temp
                A += self.log_deriv(lc[j])*temp

            OE += p[i] * np.conj(L_loc[i]) * A

        E = p@np.abs(L_loc)**2
        ox = p@np.conj(O)

        # ⟨O(E-⟨E⟩)⟩ appears to be more stable
        f = np.conj(OE) - p@np.conj(O) * E

        #### Fischer Matrix
        S2 = np.outer(ox,np.conj(ox))

        S = S1 - S2 + self.reg*np.eye(A.shape[0])
        # np.linalg.cholesky(S)

        gamma = -self.eta*np.linalg.pinv(S)@f

        return gamma, {"Mean":E, "Sigma": np.nan , "Variance": np.nan, "Rhat": np.nan}


    def log_deriv(self,c):
        O = self.vqs.ma.log_deriv(relabel(c))
        return O

    def local_cost(self):

        L = self.lind # is a local_liouvillian object

        l_cost = np.zeros(len(self.vqs.samples[0]),dtype=np.complex128)

        for i in range(len(self.vqs.samples[0])):
            Lc,Lv,k = L.get_row(self.vqs.samples[0][i])

            for j in range(k):
                #
                l_cost[i] += Lv[j] * self.vqs.ma.rho(Lc[j])  / self.vqs.samples[1][i]

        return l_cost

    def cost(self):
        # if(self.vqs.samples[0] is None): self.vqs.sampling()
        return np.real(self.local_cost()@self.probability())

    def probability(self):
        return self.vqs.probability()




class NatGradSampling():
    '''
    Caluclates the parameter updates using markov samples.

    '''
    def __init__(self,vqs,lind,eta=0.02,l=0.01):
        '''
        eta: learning rate
        reg: regularization of fisher matrix
        vqs: variational quantum state object
        lind: lindblad object
        '''

        self.eta = eta
        self.reg = l

        self.vqs = vqs #passed each step again
        self.lind = lind #should get initialised as soon as the driver gets init

    def updates(self):
        '''
        returns statistics on markov chains as well as updates
        '''
        C_stats, O, OE, OO = self.expectations()
        A = OO - np.outer(np.conj(O),O)
        S = A + self.reg*np.eye(A.shape[0])
        f = OE - np.conj(O)*C_stats["Mean"]

        # np.linalg.cholesky(S)

        gamma = self.eta*np.linalg.pinv(S)@f

        return gamma, C_stats

    def expectations(self):
        '''
        uses samples stored in vqs.
        calculates the necessary expectation values for the quantum fisher matrix
        and the gradient.

        '''

        nc = len(self.vqs.samples[0])
        ns = len(self.vqs.samples[0][0])

        o = np.zeros([nc,self.vqs.Np,ns],dtype=np.complex128)
        OO = np.zeros([nc,self.vqs.Np,self.vqs.Np],dtype=np.complex128)
        OE = np.zeros((nc,self.vqs.Np),dtype=np.complex128)

        c = self.local_cost()
        c_stats = sts.mc_stats(c)

        args = zip(c,o,OO,OE,self.vqs.samples[0])

        # calculates the expectations on four threads. adapt to number of cores
        pool = mp.Pool(4)
        results = pool.starmap(self.single_chain_expectations,args)
        pool.close()

        # recombines the separate expectations
        for i in range(nc):
            o[i] = results[i][0]
            OO[i] = results[i][1]
            OE[i] = results[i][2]

        # mean over samples
        O = np.mean(o,axis=(0,2))
        OO = np.sum(OO,axis=0)/nc
        OE = np.sum(OE,axis=0)/nc

        return c_stats, O, OE, OO

    def single_chain_expectations(self,c,o,OO,OE,samples):
        '''calculates the expectation value over a single markov chain.
        c: job number
        o, OO, OE: zero pre-allocated arrays to store expectations
        samples: samples of a single markov chain
        '''
        ns = len(samples)

        o[:,0] = self.log_deriv(samples[0])
        OO += np.outer(np.conj(o[:,0]),o[:,0]) / (ns)
        OE += (np.conj(o[:,0]) * c[0])/(ns) #expectation

        for i in range(1,ns):
            if(mcy.check_equal(samples[i-1],samples[i])):
                o[:,i] = o[:,i-1]
            else:
                o[:,i] = self.log_deriv(samples[i])

            OO += np.outer(np.conj(o[:,i]),o[:,i]) / (ns) #expectation
            OE += (np.conj(o[:,i]) * c[i])/(ns) #expectation

        return o, OO, OE


    def local_cost(self):
        '''
        estimates the local cost
        '''
        L = self.lind # is a local_liouvillian object

        nchains = len(self.vqs.samples[0])
        nsamples = len(self.vqs.samples[0][0])

        l_cost = np.zeros((nchains,nsamples),dtype=np.complex128)

        '''
        can be parallelized! chains don't talk to each other
        '''
        for h in range(nchains):
            Lc,Lv,k = L.get_row(self.vqs.samples[0][h][0])
            for j in range(k):
                l_cost[h,0] += Lv[j] * self.vqs.ma.rho(Lc[j])  / self.vqs.samples[1][h][0]

            for i in range(1,nsamples):
                if(mcy.check_equal(self.vqs.samples[0][h][i-1],self.vqs.samples[0][h][i])):
                    l_cost[h,i] = l_cost[h,i-1]
                else:
                    Lc,Lv,k = L.get_row(self.vqs.samples[0][h][i])

                    for j in range(k):
                        l_cost[h,i] += Lv[j] * self.vqs.ma.rho(Lc[j])  / self.vqs.samples[1][h][i]

        return l_cost

    def cost(self):
        if(self.vqs.samples[0] is None): self.vqs.sampling()
        return np.mean(self.local_cost())

    def probability(self):
        return self.vqs.probability()

    def log_deriv(self,c):
        O = self.vqs.ma.log_deriv(relabel(c))
        # O = self.vqs.ma.log_deriv(c)
        return O

#End


'''
Deprecated
    def expectations(self):
        ##### WHAT NEEDS TO BE DONE
        nc = len(self.vqs.samples[0])
        ns = len(self.vqs.samples[0][0])


        o = np.zeros([self.vqs.Np,nc,ns],dtype=np.complex128)
        OO = np.zeros([self.vqs.Np,self.vqs.Np],dtype=np.complex128)
        oe_test = np.zeros(self.vqs.Np,dtype=np.complex128)

        c = self.local_cost()
        c_stats = sts.mc_stats(c)

        for h in range(nc):
            o[:,h,0] = self.log_deriv(self.vqs.samples[0][h][0])
            OO[:,:] += np.outer(np.conj(o[:,h,0]),o[:,h,0]) / (nc*ns)

            for i in range(1,ns):
                if(mcy.check_equal(self.vqs.samples[0][h][i-1],self.vqs.samples[0][h][i])):
                    o[:,h,i] = o[:,h,i-1]
                    # OO[:,:] += np.outer(np.conj(o[:,i]),o[:,i]) / self.vqs.ns
                else:
                    o[:,h,i] = self.log_deriv(self.vqs.samples[0][h][i])
                OO[:,:] += np.outer(np.conj(o[:,h,i]),o[:,h,i]) / (nc*ns)

        O = np.mean(o,axis=(1,2))
        OE = np.mean(np.conj(o.reshape(self.vqs.Np,nc*ns)) @ np.diag(np.concatenate(c)),1)


        return c_stats, O , OE, OO



    def expectations2(self):
        ##### WHAT NEEDS TO BE DONE
        nc = len(self.vqs.samples[0])
        ns = len(self.vqs.samples[0][0])

        #
        o = np.zeros([nc,self.vqs.Np,ns],dtype=np.complex128)
        OO = np.zeros([nc,self.vqs.Np,self.vqs.Np],dtype=np.complex128)
        OE = np.zeros((nc,self.vqs.Np),dtype=np.complex128)

        #cost
        c = self.local_cost()
        c_stats = sts.mc_stats(c)

        def single_chain_expectations(h,c,o,OO,OE,samples):
            samples: only the configs! (I should really dict the whole shit)

            every input is h dependent and has h in first index.
            starmap should (test properly) take the first index of each object
            and pass it to

            nc = len(samples)
            ns = len(samples[0])

            o[h,:,0] = self.log_deriv(samples[h][0])
            OO[h] += np.outer(np.conj(o[h,:,0]),o[h,:,0]) / (nc*ns)
            OE[h] += (np.conj(o[h,:,0]) * c[h,0])/(nc*ns) #expectation

            for i in range(1,ns):
                if(mcy.check_equal(samples[h][i-1],samples[h][i])):
                    o[h,:,i] = o[h,:,i-1]
                else:
                    o[h,:,i] = self.log_deriv(samples[h][i])

                OO[h] += np.outer(np.conj(o[h,:,i]),o[h,:,i]) / (nc*ns) #expectation
                OE[h] += (np.conj(o[h,:,i]) * c[h,i])/(nc*ns) #expectation

            return o[h], OO[h], OE[h]

        for h in range(nc):
            o[h], OO[h], OE[h] = single_chain_expectations(h,c,o,OO,OE,self.vqs.samples[0])



        OO = np.sum(OO,axis=0)
        OE = np.sum(OE,axis=0)
        O = np.mean(o,axis=(0,2))


        return c_stats, O, OE, OO

    def expectations(self):
        ##### WHAT NEEDS TO BE DONE
        nc = len(self.vqs.samples[0])
        ns = len(self.vqs.samples[0][0])

        #
        o = np.zeros([self.vqs.Np,nc,ns],dtype=np.complex128)
        OO = np.zeros([self.vqs.Np,self.vqs.Np],dtype=np.complex128)
        OE = np.zeros(self.vqs.Np,dtype=np.complex128)

        #cost
        c = self.local_cost()
        c_stats = sts.mc_stats(c)

        for h in range(nc):
            o[:,h,0] = self.log_deriv(self.vqs.samples[0][h][0])
            OO += np.outer(np.conj(o[:,h,0]),o[:,h,0]) / (nc*ns)
            OE += (np.conj(o[:,h,0]) * c[h,0])/(nc*ns) #expectation

            for i in range(1,ns):
                if(mcy.check_equal(self.vqs.samples[0][h][i-1],self.vqs.samples[0][h][i])):
                    o[:,h,i] = o[:,h,i-1]
                    # OO[:,:] += np.outer(np.conj(o[:,i]),o[:,i]) / self.vqs.ns
                else:
                    o[:,h,i] = self.log_deriv(self.vqs.samples[0][h][i])

                OO += np.outer(np.conj(o[:,h,i]),o[:,h,i]) / (nc*ns) #expectation
                OE += (np.conj(o[:,h,i]) * c[h,i])/(nc*ns) #expectation


        O = np.mean(o,axis=(1,2))


        return c_stats, O, OE, OO



    def local_cost_LL(self):
        L = self.lind # is a local_liouvillian object

        nchains = len(self.vqs.samples[0])
        nsamples = len(self.vqs.samples[0][0])

        l_cost = np.zeros((nchains,nsamples),dtype=np.complex128)


        can be parallelized! chains don't talk to each other

        for h in range(nchains):
            Lc,Lv,k = L.get_conn_LL(self.vqs.samples[0][h][0])
            for j in range(k):
                l_cost[h,0] += Lv[j] * self.vqs.ma.rho(Lc[j])  / self.vqs.samples[1][h][0]

            for i in range(1,nsamples):
                if(mcy.check_equal(self.vqs.samples[0][h][i-1],self.vqs.samples[0][h][i])):
                    l_cost[h,i] = l_cost[h,i-1]
                else:
                    Lc,Lv,k = L.get_conn_LL(self.vqs.samples[0][h][i])

                    for j in range(k):
                        l_cost[h,i] += Lv[j] * self.vqs.ma.rho(Lc[j])  / self.vqs.samples[1][h][i]


        return l_cost
'''
