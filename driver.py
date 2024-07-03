import numpy as np
import time
import utility as ut

class steady_state():
    '''
    Class which run the optimizer and updates the machine
    '''
    def __init__(self, optimizer):
        '''
        optimizer: NatGradSampling or NatGradExact object
        obsservable: local_observable object. If passed will calculate expectation at every iteration (slow)
        '''

        self.op = optimizer
        self.cost = None    # gets updated
        self.stats_history = {}
        self.epoch = 0


    def cost_func(self):
        self.cost = self.op.cost()
        return self.cost

    def rho(self,c):
        return self.op.vqs.ma.rho(c)

    def state(self):
        self.op.vqs.vs_state()

    def _update_cost(self,c_stats):
        self.cost = c_stats["Mean"]
        self.stats_history[self.epoch] = c_stats

    def _update_epoch(self):
        self.epoch += 1

    def expectation(self,observable):
        '''
        observable: local_observable object
        returns: expectation value
        '''
        # self.op.vqs.diag_sampling()
        o = self.op.vqs.expectation(observable)
        return o

    # def cosine_schedule(self,Niter,reg_max,i,cosine_decay = 1):
    #     reg_min = cosine_decay * reg_max
    #
    #     x = 0.5*np.pi/Niter
    #     return max(reg_max*np.cos(x*i),reg_min)

    def exponential_schedule(self,i,a0=0.02,amin=1e-4,b=0.999):
        return max(a0*(b**i),amin)

    def run(self, niter = None, cut = 1e-4, Verbose = False, log_full_state = False, p_log=None):
        '''
        niter: Number of iterations to run
        cut: If no niter, otimize until LdagL < cut
        Verbose: If verbose, print current step and LL
        iteratively optimizes the variational state.
        optionally saves parameters and cost at every step
        '''

        E = 10
        i = 0

        total_time = 0
        run_avg = 0

        if log_full_state:
            state_hist = np.zeros((niter,2**self.op.vqs.Nv,2**self.op.vqs.Nv),dtype=np.complex128)

        for i in range(niter):
            # self.op.eta = self.exponential_schedule(i,a0=1e-2,amin=3e-3,b=0.999)
            t0 = time.time()

            self.op.vqs.reset()
            self.op.vqs.sampling()

            updates, c_stats = self.op.updates()

            self._update_epoch()
            self._update_cost(c_stats)

            self.op.vqs.update(updates)

            t1 = time.time()

            time_delta = (t1-t0)
            total_time += time_delta
            run_avg = total_time/(i+1)

            if log_full_state is True:
                state_hist[i] = self.op.vqs.vs_state()

            # self.op.l = c_stats["Variance"]

            if not p_log is None:
                if(i % p_log == 0):
                    np.save(f"plog_{i}",self.op.vqs.ma.params())


            if(Verbose):
                print(f' Epoch: {i+1}/{niter}, <s/it>: {time_delta:.2}s, ETC: {int(run_avg*(niter-(i+1)))}s, |<L>|²: {np.abs(c_stats["Mean"])**2:.3}, [\u03C3: \u00B1{ c_stats["Sigma"]:.3}, V: {c_stats["Variance"]:.3} R: {c_stats["Rhat"]:.3}]',end='\r')
            else:
                print(f' Epoch: {i+1}/{niter}, <s/it>: {run_avg:.2}s, ETC: {int(total_time)}s < {int(run_avg*(niter-(i+1)))}s', end='\r')

        if log_full_state:
            np.save("state_hist",state_hist)

#End
