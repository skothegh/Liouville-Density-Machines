import numpy as np

def mc_stats(batch):

    mean = _mean(batch)
    rhat = np.abs(_grubin(batch))
    sem = _sem(batch)
    variance = _var(batch)

    return {"Mean": mean, "Sigma": sem, "Variance": variance, "Rhat": rhat }

def _sem(batch):
    nc,ns = batch.shape
    return np.std(batch,ddof=1)/np.sqrt(nc*ns)

def _grubin(batch):
    '''
    calculates the simple (1992) Gelman-Rubin R-hat
    should be smaller than ~1.02
    '''
    nchains, nsamples = batch.shape

    B = 0
    chain_variance = np.zeros(nchains,dtype=np.complex128)

    chain_mean = np.mean(batch,axis=1)
    grand_mean = np.mean(chain_mean)

    for i in range(nchains):
        B += (nsamples/(nchains-1)) * (chain_mean[i] - grand_mean)**2

    for i in range(nchains):
        for j in range(nsamples):
            chain_variance[i] += (1/(nsamples-1)) * (batch[i,j] - chain_mean[i])**2

    W = np.mean(chain_variance)

    if(not np.abs(W) < 1e-15):
        R = (nsamples-1)/nsamples + B/(nsamples*W)
    elif(np.abs(W) < 1e-15 and np.abs(B) <1e-15):
        R = 1
    else:
        '''
        i care for the cases where this fails
        '''
        R = np.nan
        print(W)

    return np.sqrt(R)

def _mean(batch):
    return np.mean(batch)

def _var(batch):
    return np.var(batch,ddof=1)

def _is_good_batch(Rhat):
    '''
    should include a measure of the autocorrelation as well as
    the standard monte carlo error. latter is unit-ful and needs to
    be taken in relation to the mean. if all are ok the batch is good
    -> calc updates otherwise use chains to keep sampling
    '''
    return Rhat < 1.02
