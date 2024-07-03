import cython
import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array as cvarray
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.time cimport time,time_t
# from functools import lru_cache

########## Utility ##########
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef check_equal(long[:] arr1, long[:] arr2):
    cdef int v = 1
    cdef int i, n1, n2

    n1 = len(arr1)
    n2 = len(arr2)

    assert n1 == n2

    for i in range(n1):
      if(not arr1[i] == arr2[i]):
        v *= 0
        break

    return v

@cython.boundscheck(False)
@cython.wraparound(False)
cdef copy_cy(long[:] arr):

    cdef int n = len(arr)

    temp = cvarray(shape=(n,), itemsize=sizeof(long), format="l")
    # cdef np.ndarray[np.int64_t, ndim=1] temp = np.zeros(n, dtype=np.int64)
    cdef long[:] t_view = temp

    cdef int i

    for i in range(n):
      t_view[i] = arr[i]calcualates derivatives of

    return t_view

########## Special Definitions ##########

cdef extern from "math.h":
    double exp(double)
    long sqrt(long)

cdef extern from "complex.h":
    double complex cexp(double complex)
    long complex I
    double complex ccosh(double complex)
    double complex conj(double complex)
    double complex ctanh(double complex)
    double complex cabs(double complex)


from libc.stdlib cimport srand, rand, RAND_MAX

cdef bint boolean_variable = True

@cython.cdivision(True)
cpdef int crandint(int lower, int upper) except -1:
    return (rand() % (upper - lower)) + lower

@cython.boundscheck(False)
@cython.wraparound(False)
cdef crandarray(int sites, int max):
    cdef np.ndarray[np.int64_t, ndim = 1] arr = np.empty(sites, dtype=np.int64)
    cdef long[:] arr_view = arr

    cdef int i

    for i in range(sites):
      arr_view[i] = crandint(0, max)

    return arr_view

@cython.cdivision(True)
cpdef float crandom() except -1:
    return  <float>rand() / <float>RAND_MAX


cdef get_time():
    cdef timespec ts
    # cdef double current
    clock_gettime(CLOCK_REALTIME, &ts)
    return ts.tv_sec + (ts.tv_nsec / 1000000000.)

    # cdef time_t t = time(NULL)

    # return t
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef get_seed(int num):
    srand(num*((get_time()%1)*100000))


########## Machine ##########
# cdef relabel(long[:] c0):
#     cnew = cvarray(shape=(len(c0),), itemsize=sizeof(long), format="l")
#
#     cdef int i
#     cdef long[:] c_view = cnew
#
#
#     for i in range(len(c0)):
#         if(c0[i] == 0):
#           c_view[i] = 2
#         elif(c0[i] == 1):
#           c_view[i] = 1
#         elif(c0[i] == 2):
#           c_view[i] = -1
#         elif(c0[i] == 3):
#           c_view[i] = -2
#
#
#     return cnew


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef log_deriv_cy(int Nv,
                   int Nh,
                   complex[:] b,
                   complex[:,::1] u,
                   complex[:,::1] v,
                   complex[:,::1] w,
                   long[:] c):

    cdef int Np = 3*Nv + Nh + 3*Nv*Nh

    # cdef long[:] c = relabel(c0)

    O = 1j*np.zeros(Np,dtype=np.complex128)
    cdef complex[:] O_view = O

    #cdef complex[:] cc = complex_casting(c)
    cdef complex temp
    cdef int i, j, k
    for j in range(Nh): #b derivative
      temp = 0.0
      for k in range(Nv): # matrix multiplication (like i do in local_cost)
        temp += u[j,k]*c[k] + v[j,k]*c[k]*c[k] + w[j,k]*c[k]*c[k]*c[k]
      O_view[3*Nv + j] = ctanh(b[j] + temp)

    for i in range(Nv):
        # a deriv
        O_view[i] = c[i]    #a1 derivative
        O_view[Nv + i] = c[i]*c[i]  # a2 derivative
        O_view[2*Nv + i] = c[i]*c[i]*c[i] # a3 derivative

        for j in range(Nh): # should be u,v,w derivative
            O_view[3*Nv + Nh +  Nv*j + i ] = c[i] * O_view[3*Nv + j]
            O_view[3*Nv + (1+Nv)*Nh + Nv*j + i ] = c[i]*c[i] * O_view[3*Nv + j]
            O_view[3*Nv + (1+2*Nv)*Nh + Nv*j + i ] = c[i]*c[i]*c[i] * O_view[3*Nv + j]

    # print(np.imag(O))
    return O

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rho_cy(complex[:] a1,
             complex[:] a2,
             complex[:] a3,
             complex[:] b,
             complex[:,:] u,
             complex[:,:] v,
             complex[:,:] w,
             long[:] c0):

    cdef int i,j,k
    cdef complex temp1, temp2, temp3, s

    # cdef np.ndarray[np.int64_t, ndim=1] c = np.empty(len(c0),dtype=np.int64)
    c = cvarray(shape=(len(c0),), itemsize=sizeof(long), format="l")

    cdef long[:] c_view = c

    for i in range(len(c0)):
      if(c0[i] == 0):
        c_view[i] = 2
      elif(c0[i] == 1):
        c_view[i] = 1
      elif(c0[i] == 2):
        c_view[i] = -1
      elif(c0[i] == 3):
        c_view[i] = -2

    # print(c_view[0],c_view[1])

    temp1 = 0.0
    temp2 = 0.0
    temp3 = 0.0

    for i in range(len(c_view)):
      temp1 += a1[i] * c_view[i]
      temp2 += a2[i] * c_view[i] * c_view[i]
      temp3 += a3[i] * c_view[i] * c_view[i] * c_view[i]

    s = cexp(temp1 + temp2 + temp3)

    for j in range(len(b)):
      temp1 = 0.0
      temp2 = 0.0
      temp3 = 0.0
      for k in range(c_view.shape[0]):
        temp1 += u[j,k] * c_view[k]
        temp2 += v[j,k] * c_view[k] * c_view[k]
        temp3 += w[j,k] * c_view[k] * c_view[k] * c_view[k]

      s *= 2 * ccosh(b[j] + temp1 + temp2 + temp3)
      # s *= ccosh(b[j] + temp1 + temp2 + temp3)

    return s


########## Liouville ##########


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef config_cy(int Nv,
                int i):

    cdef long d = 4 ** Nv
    cdef int ds = int(d // 4)

    config = np.zeros(Nv,dtype=np.int64)
    cdef long[:] config_view = config

    cdef int j

    for j in range(Nv):
        config_view[j] = (i % (d / 4**(j) )) //  (ds // 4**j)

    return config

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef idx_cy(int Nv,
              long[:] config):

    cdef double idx = 0
    cdef int i

    for i in range(Nv):
        idx += config[i] * 4**(Nv - 1 - i)
    return int(idx)


########## Variational Quantum State ##########
# cdef idx_rho_cy(complex[:] a1,
#                  complex[:] a2,
#                  complex[:] a3,
#                  complex[:] b,
#                  complex[:,:] u,
#                  complex[:,:] v,
#                  complex[:,:] w,
#                  int Nv,
#                  int j):
#     return rho_cy(a1,a2,a3,b,u,v,w,config_cy(Nv,j)) # this will later return a "new configuration..." might keep it for the background?


########## Markov Chain Monte Carlo ##########
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef burn_in(complex[:] a1,
              complex[:] a2,
              complex[:] a3,
              complex[:] b,
              complex[:,:] u,
              complex[:,:] v,
              complex[:,:] w,
              int Nsites,
              int burn_in_period = 100):
    # srand(time())
    # get_seed(seed)

    cdef int i
    cdef float p
    cdef complex s0, si

    cdef long[:] ci
    cdef long[:] c0 = crandarray(Nsites,4)

    s0 = rho_cy(a1,a2,a3,b,u,v,w,c0)

    for i in range(burn_in_period):
        ci = single_jump_cy(Nsites, c0)
        si = rho_cy(a1,a2,a3,b,u,v,w,ci)

        p = float(cabs(si/s0)**2)

        if(crandom() < p):
          c0 = ci
          s0 = si

    return c0, s0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef diag_burn_in(complex[:] a1,
              complex[:] a2,
              complex[:] a3,
              complex[:] b,
              complex[:,:] u,
              complex[:,:] v,
              complex[:,:] w,
              int Nsites,
              int burn_in_period = 100):


    cdef int i
    cdef float p
    cdef complex s0, si

    cdef long[:] ci
    cdef long[:] c0 = crandarray(Nsites,2)
    cdef np.ndarray[np.int64_t, ndim = 1] ct = np.empty(Nsites, dtype=np.int64)

    s0 = rho_cy(a1,a2,a3,b,u,v,w,c0)

    for i in range(burn_in_period):
        ci = diag_jump(Nsites, c0)
        for j in range(Nsites):
          ct[j] = ci[j] * 3

        si = rho_cy(a1,a2,a3,b,u,v,w,ct)

        p = float(cabs(si/s0))

        if(crandom() < p):
          c0 = ci
          s0 = si

    return c0, s0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef diag_mcmc(complex[:] a1,
                complex[:] a2,
                complex[:] a3,
                complex[:] b,
                complex[:,:] u,
                complex[:,:] v,
                complex[:,:] w,
                int Nv,
                int ns,
                int nchains
                ):

    cdef int i,j,k

    cdef np.ndarray[np.int64_t, ndim = 3] config = np.zeros((nchains,ns,Nv),dtype=np.int64)
    cdef long[:,:,:] c_view = config

    cdef np.ndarray[np.complex128_t, ndim = 2] sts = np.zeros((nchains,ns),dtype=np.complex128)
    cdef complex[:,:] sts_view = sts

    cdef long[:] rt, ct

    cdef complex st
    cdef float beta = 1.0

    cdef int A, a
    # A = 0

    # for i in range(nchains):
    #     rt = crandarray(Nv, 2)
    #     c_view[i,0] = rt
    #     for j in range(Nv):
    #       ct2[i,j] = c_view[i,0,j] * 3
    #
    #     sts_view[i,0] = rho_cy(a1,a2,a3,b,u,v,w,ct2[i])

    for i in range(nchains):
      ct, st = diag_burn_in(a1,a2,a3,b,u,v,w,Nv)
      c_view[i,0,:] = ct
      sts_view[i,0] = st

    for i in range(nchains):
        for j in range(1,ns):
            ct, st, a = diag_sweep(a1,a2,a3,b,u,v,w,Nv,config[i,j-1],sts_view[i,j-1])
            c_view[i,j,:] = ct
            sts_view[i,j] = st
            # A+=a

    for i in range(nchains):
      for j in range(ns):
        for k in range(Nv):
          c_view[i,j,k] = c_view[i,j,k]*3
    #
    return [config, sts]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cpdef multi_mcmc_cy(complex[:] a1,
              complex[:] a2,
              complex[:] a3,
              complex[:] b,
              complex[:,:] u,
              complex[:,:] v,
              complex[:,:] w,
              int Nv,
              int nsamples,
              int nchains,
              int nsweeps):

    cdef int i,j
    cdef float beta = 1.0

    cdef np.ndarray[np.int64_t, ndim = 3] config = np.zeros((nsamples,nchains,Nv),dtype=np.int64)
    cdef long[:,:,:] c_view = config

    cdef np.ndarray[np.complex128_t, ndim = 2] sts = np.zeros((nsamples,nchains),dtype=np.complex128)
    cdef complex[:,:] sts_view = sts

    cdef long[:] rt, ct
    cdef complex st

    for i in range(nchains):
        rt = crandarray(Nv, 4)
        c_view[0,i] = rt
        sts_view[0,i] = rho_cy(a1,a2,a3,b,u,v,w,c_view[0,i])

    for i in range(1,nsamples):
        for j in range(nchains):
            ct, st = mcmc_sweep(a1,a2,a3,b,u,v,w,config[i-1,j],sts_view[i-1,j],Nv,nsweeps)
            c_view[i,j] = ct
            sts_view[i,j] = st

    return config, sts

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
cpdef single_chain_cy(complex[:] a1,
              complex[:] a2,
              complex[:] a3,
              complex[:] b,
              complex[:,:] u,
              complex[:,:] v,
              complex[:,:] w,
              int Nsites,
              int Nsamples,
              int Nsweeps,
              int seed,
              long[:,:] Configs,
              complex[:] States):

    #pass
    get_seed(seed)
    cdef int j = 1
    cdef long[:] ct
    cdef complex st
    cdef float alpha = 0.0

    for j in range(1,Nsamples):
    # while(alpha < Nsamples):
      # print(j)
      ct,st = mcmc_sweep(a1,a2,a3,b,u,v,w,Configs[j-1],States[j-1],Nsites,Nsweeps)
      # print(j,a)

      Configs[j] = ct
      States[j] = st

      # alpha += a
      # j += 1

    return np.array(Configs), np.array(States)


@cython.cdivision(True)
cdef mcmc_sweep(complex[:] a1,
              complex[:] a2,
              complex[:] a3,
              complex[:] b,
              complex[:,:] u,
              complex[:,:] v,
              complex[:,:] w,
              long[:] cold,
              complex sold,
              int Nv,
              int nsweeps
              ):



    cdef int i
    cdef long[:] temp = copy_cy(cold)
    cdef long[:] cnew

    cdef complex snew
    cdef float p
    cdef float a = 0.0
    cdef int j = 1
    for i in range(nsweeps):
    # while(a < nsweeps):

        cnew = single_jump_cy(Nv,temp)
        snew = rho_cy(a1,a2,a3,b,u,v,w,cnew)

        p = float(cabs((snew/sold))**2)

        if(crandom() < p):
            a += 1.0
            sold = snew
            temp = cnew
    return temp, sold


@cython.cdivision(True)
cdef diag_sweep(complex[:] a1,
               complex[:] a2,
               complex[:] a3,
               complex[:] b,
               complex[:,:] u,
               complex[:,:] v,
               complex[:,:] w,
               int Nv,
               long[:] config_old,
               complex state_old,
               ):


    '''
    Samples from real, classical probability distribution -> does not need the |a/b|^2

    '''
    cdef int i,j
    cdef complex snew, stemp

    cdef long[:] temp = copy_cy(config_old)
    cdef long[:] cnew
    cdef long[:] con
    cdef float p

    stemp = state_old

    ct = cvarray(shape=(Nv,), itemsize=sizeof(long), format="l")

    N = 0

    for j in range(20):
        # print(j)
        cnew = diag_jump(Nv,temp) # returns a "new config"

        for i in range(Nv):
          ct[i] = cnew[i]*3

        snew = rho_cy(a1,a2,a3,b,u,v,w,ct) # needs to relabel the "new config" still rho2 since both require relabel!

        p = float(cabs(snew/stemp))
        # print([cnew[0],cnew[1],cnew[2],snew],[temp[0],temp[1],temp[2],state_old],p)
        # print(index,state_old)
        if(crandom() < p):
            N+=1
            # print("s")
            temp = cnew
            stemp = snew

    return temp, stemp, N

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef diag_jump(int Nv,
                long[:] start):

      cdef long[:] cn = copy_cy(start)
      cdef int i = crandint(0,Nv)
      cn[i] = (start[i] + 1) % 2

      return cn


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef long[:] single_jump_cy(int Nv,
                             long[:] start
                             ):

    cdef int i,j
    cdef long[:] end = copy_cy(start)

    i = crandint(0,Nv)
    # print(start[0],start[1])
    j = crandint(0,3)

    end[i] = (end[i] + j + 1) % 4

    return end


########## Natural Gradient ##########
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef local_cost_cy(complex[:] a1,
                   complex[:] a2,
                   complex[:] a3,
                   complex[:] b,
                   complex[:,:] u,
                   complex[:,:] v,
                   complex[:,:] w,
                   int Nv,
                   complex[:,:] lind,
                   int[:] indices,
                   complex[:] samples,
                   func):

    cdef int ns = len(samples)
    cdef int d0 = lind.shape[0]
    cdef int d1 = lind.shape[1]

    cdef np.ndarray LL = np.zeros((d0,d1),dtype=np.complex128)
    cdef complex[:,:] LL_view = LL

    cdef np.ndarray lc = np.zeros(ns,dtype=np.complex128)
    cdef complex[:] lc_view = lc

    cdef int i,j,k

    for i in range(d0):
      for j in range(d1):
        for k in range(d1):
          LL_view[i,j] += conj(lind[k,i])*lind[k,j]

    cdef long[:] n0

    for i in range(ns):
      n0 = LL[indices[i]].nonzero()[0]
      for j in n0:
        lc_view[i] += LL_view[indices[i],j] * func(j) / samples[i]

    return lc




########## Liouvillian Class ##########
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef get_conn_LL(long[:] config,
                   int ns,
                   complex[:,:,:] gates,
                   complex[:,:,:] Tgates):

    cdef long[:,:] Ldc, Lc
    cdef complex[:] Ldv, Lv

    cdef int Ldk, Lk, k, j, f
    cdef int os = gates.shape[0]

    Ldc, Ldv, Ldk = list_col_cons(Tgates,
                                  config,
                                  ns,
                                  conjugate = True) #Ld connections

    cdef int  LLf = 0

    cdef np.ndarray[np.int64_t, ndim = 2] LLc = np.empty((16 * 16 * os,ns), dtype=np.int64)
    cdef long[:,:] LLc_view = LLc

    cdef np.ndarray[np.complex128_t, ndim = 1] LLv = np.zeros(16 * 16 * os, dtype=np.complex128)
    cdef complex[:] LLv_view = LLv


    for k in range(Ldk): # number of Ld connections
        Lc, Lv, Lk = list_row_cons(gates,
                                   Ldc[k],
                                   ns) # now i have the "ith j"

        for j in range(Lk):
            for f in range(LLf+1):
                if(f == LLf):
                    LLv_view[f] += Lv[j] * Ldv[k]
                    LLc_view[f] = Lc[j]
                    LLf += 1

                elif(check_equal(Lc[j],LLc[f])):
                    LLv_view[f] += Lv[j] * Ldv[k]
                    break

    return LLc, LLv, LLf

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cpdef get_col(complex[:,:,:] gates,
              long[:] config,
              int ns,
              bint conjugate):


    cdef long[:,:] Lc
    cdef complex[:] Lv
    cdef int Lk

    Lc, Lv, Lk = list_col_cons(gates,
                                  config,
                                  ns,
                                  conjugate = False) #Ld connections

    cdef int i, j, k
    cdef int os = gates.shape[0]

    cdef np.ndarray[np.int64_t, ndim = 2] ColC = np.empty((16 * os,ns), dtype=np.int64)
    cdef long[:,:] ColC_view = ColC

    cdef np.ndarray[np.complex128_t, ndim = 1] ColV = np.zeros(16 * os, dtype=np.complex128)
    cdef complex[:] ColV_view = ColV

    k = 0
    for i in range(Lk):
        for j in range(k+1):
            if(j == k):
                ColV_view[j] += Lv[j]
                ColC_view[j] = Lc[j]
                k += 1
            elif(check_equal(Lc[i],ColC[j])):
                ColV_view[j] += Lv[i]
                break

    return ColC, ColV, k

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cpdef list_col_cons(complex[:,:,:] gates,
                        long[:] config,
                        int n,
                        bint conjugate):

    cdef int i,j,k
    cdef int os = gates.shape[0]

    cdef np.ndarray[np.complex128_t, ndim = 1] Lv = np.empty(16 * os, dtype=np.complex128)
    cdef complex[:] Lv_view = Lv

    cdef np.ndarray[np.int64_t, ndim = 2] Lc = np.empty((16 * os,n), dtype=np.int64)
    cdef long[:,:] Lc_view = Lc

    cdef np.ndarray temp = np.empty(2,dtype=np.int64)
    cdef long[:] t_view = temp

    cdef np.ndarray temp2 = np.empty(2,dtype=np.int64)
    cdef long[:] t2_view = temp2

    cdef long[:] ct
    cdef complex[:] col

    k = 0
    for i in range(os):
        ct = copy_cy(config)

        t_view[0] = config[i]
        t_view[1] = config[i+1]

        col = gates[i][idx_cy(2,temp)]
        for j in range(16):
            if(not col[j] == 0):
                t2_view = config_cy(2,j)
                ct[i] = t2_view[0]
                ct[i+1] = t2_view[1]

                Lc_view[k] = ct
                if(conjugate):
                  Lv_view[k] = conj(col[j])
                else:
                  Lv_view[k] = col[j]
    #
                k += 1
    #
    return Lc,Lv, k



@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cpdef get_row(complex[:,:,:] gates,
              long[:] config,
              int ns
              ):

    pass
    cdef long[:,:] Lc
    cdef complex[:] Lv
    cdef int Lk
#
    Lc, Lv, Lk = list_row_cons(gates,
                              config,
                              ns) # now i have the "ith j"
#
    cdef int i, j, k
    cdef int os = gates.shape[0]

    cdef np.ndarray[np.int64_t, ndim = 2] RowC = np.empty((16 * os,ns), dtype=np.int64)
    cdef long[:,:] RowC_view = RowC

    cdef np.ndarray[np.complex128_t, ndim = 1] RowV = np.zeros(16 * os, dtype=np.complex128)
    cdef complex[:] RowV_view = RowV

    k = 0
    for i in range(Lk):
        for j in range(k+1):
            if(j == k):
                RowV_view[j] += Lv[i]
                RowC_view[j] = Lc[i]
                k += 1
            elif(check_equal(Lc[i],RowC[j])):
                RowV_view[j] += Lv[i]
                break

    return RowC, RowV, k


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cpdef list_row_cons(complex[:,:,:] gates,
                        long[:] config,
                        int n
                        ):


    cdef int i,j,k
    cdef int os = gates.shape[0]


    cdef np.ndarray[np.complex128_t, ndim = 1] Lv = np.empty(16 * os, dtype=np.complex128)
    cdef complex[:] Lv_view = Lv

    cdef np.ndarray[np.int64_t, ndim = 2] Lc = np.empty((16 * os,n), dtype=np.int64)
    cdef long[:,:] Lc_view = Lc

    cdef np.ndarray temp = np.empty(2,dtype=np.int64)
    cdef long[:] t_view = temp

    cdef np.ndarray temp2 = np.empty(2,dtype=np.int64)
    cdef long[:] t2_view = temp2

    cdef long[:] ct
    cdef complex[:] row

    k = 0
    for i in range(os):
        ct = copy_cy(config)

        t_view[0] = config[i]       #this is where i need to
        t_view[1] = config[i+1]     #use the
    #
        row = gates[i][idx_cy(2,temp)]
        for j in range(16):
            if(not row[j] == 0):

                t2_view = config_cy(2,j)
                ct[i] = t2_view[0]
                ct[i+1] = t2_view[1]

                Lc_view[k] = ct
                Lv_view[k] = row[j]
    #
                k += 1

    return Lc,Lv, k




@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cpdef get_row_new(int nsites,
              int[:,:] bonds,
              complex[:,:,:] gates,
              long[:] config):

    cdef long[:,:] Lc
    cdef complex[:] Lv
    cdef int Lk
#
    Lc, Lv, Lk = list_row_cons_new(nsites,
                               bonds,
                               gates,
                               config)

    cdef int i, j, k
    cdef int os = gates.shape[0]

    cdef np.ndarray[np.int64_t, ndim = 2] RowC = np.empty((16 * os,nsites), dtype=np.int64)
    cdef long[:,:] RowC_view = RowC

    cdef np.ndarray[np.complex128_t, ndim = 1] RowV = np.zeros(16 * os, dtype=np.complex128)
    cdef complex[:] RowV_view = RowV

    k = 0
    for i in range(Lk):
        for j in range(k+1):
            if(j == k):
                RowV_view[j] += Lv[i]
                RowC_view[j] = Lc[i]
                k += 1
            elif(check_equal(Lc[i],RowC[j])):
                RowV_view[j] += Lv[i]
                break

    return RowC, RowV, k


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cpdef list_row_cons_new(int nsites,
                    int[:,:] bonds,
                    complex[:,:,:] gates,
                    long[:] config):


    cdef int i,j,k
    cdef int os = gates.shape[0]


    cdef np.ndarray[np.complex128_t, ndim = 1] Lv = np.empty(16 * os, dtype=np.complex128)
    cdef complex[:] Lv_view = Lv

    cdef np.ndarray[np.int64_t, ndim = 2] Lc = np.empty((16 * os,nsites), dtype=np.int64)
    cdef long[:,:] Lc_view = Lc

    cdef np.ndarray temp = np.empty(2,dtype=np.int64)
    cdef long[:] t_view = temp

    cdef np.ndarray temp2 = np.empty(2,dtype=np.int64)
    cdef long[:] t2_view = temp2

    cdef long[:] ct
    cdef complex[:] row

    k = 0
    for i in range(os):
        ct = copy_cy(config)

        t_view[0] = config[bonds[i,0]]       #this is where i need to
        t_view[1] = config[bonds[i,1]]     #use the bonds!
    #
        row = gates[i][idx_cy(2,temp)]
        for j in range(16):
            if(not row[j] == 0):

                t2_view = config_cy(2,j)
                ct[bonds[i,0]] = t2_view[0]
                ct[bonds[i,1]] = t2_view[1]

                Lc_view[k] = ct
                Lv_view[k] = row[j]
    #
                k += 1

    return Lc,Lv, k

#END



# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cpdef pt_mcmc_cy3(complex[:] a1,
#               complex[:] a2,
#               complex[:] a3,
#               complex[:] b,
#               complex[:,:] u,
#               complex[:,:] v,
#               complex[:,:] w,
#               int Nv,
#               int ns):
#
#       cdef int nchains = 2
#       cdef int i,j,k
#
#       indices = np.zeros((ns,nchains),dtype=np.int32)
#       cdef int[:,:] idx_view = indices
#
#       states = np.zeros((ns,nchains),dtype=np.complex128)
#       cdef complex[:,:] state_view = states
#
#       for i in range(nchains):
#         idx_view[0,i] = crandint(0,4**Nv)
#
#       # we are only relabeling. the same index still produces the same rho. once we reshuffle, thinks get funky!
#       for i in range(nchains):
#           state_view[0,i] = idx_rho_cy3(a1,a2,a3,b,u,v,w,Nv,idx_view[0,i])
#
#       cdef float beta, p, r
#
#       cdef int tint
#       cdef complex tcomp
#
#       cdef int Ns = 1
#       for i in range(1,ns,1):
#           for j in range(nchains):
#             # beta = (nchains - j)/nchains
#               beta = exp(-j/nchains)
#
#               idx_view[i,j], state_view[i,j] = mcmc_step_cy3(a1,a2,a3,b,u,v,w,Nv,idx_view[i-1,j],state_view[i-1,j],beta)
#
#           Ns+=1
#           if(Ns == ns//4):
#               for k in range(1,nchains-1,2):
#                   p = float(cabs((state_view[i,k]/state_view[i,k+1])**(1/nchains))**2)
#                   r = crandom()#np.random.uniform(0,1,1)
#
#                   if(r<p):
#                       tint = idx_view[i,k]
#                       idx_view[i,k] = idx_view[i,k+1]
#                       idx_view[i,k+1] = tint
#
#                       tcomp = state_view[i,k]
#                       state_view[i,k] = state_view[i,k+1]
#                       state_view[i,k+1] = tcomp
#
#               for k in range(0,nchains-1,2):
#                   p = float(cabs((state_view[i,k]/state_view[i,k+1])**(1/nchains))**2)
#                   r = crandom()#np.random.uniform(0,1,1)
#
#                   if(r<p):
#                       tint = idx_view[i,k]
#                       idx_view[i,k] = idx_view[i,k+1]
#                       idx_view[i,k+1] = tint
#
#                       tcomp = state_view[i,k]
#                       state_view[i,k] = state_view[i,k+1]
#                       state_view[i,k+1] = tcomp
#
#               Ns = 0
#       return [indices[:,0],states[:,0]]
