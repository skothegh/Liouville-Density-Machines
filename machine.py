import numpy as np
import machine_cy as mcy

class base_machine():
    # rbm etc should inherit from this class
    pass

class rbm(): #machine
    '''
    liouville density machine. has hardcoded parameter and derivatives.

    '''
    def __init__(self, *, visible, beta=1):
        self.Nv = visible
        self.Nh = int(visible * beta)

        self.Np = 3*self.Nv + self.Nh + 3*self.Nv*self.Nh

        self.a1 = np.zeros([self.Nv],dtype=np.complex128)
        self.a2 = np.zeros([self.Nv],dtype=np.complex128)
        self.a3 = np.zeros([self.Nv],dtype=np.complex128)

        self.b = np.zeros([self.Nh],dtype=np.complex128)

        self.u = np.zeros([self.Nh,self.Nv],dtype=np.complex128)
        self.v = np.zeros([self.Nh,self.Nv],dtype=np.complex128)
        self.w = np.zeros([self.Nh,self.Nv],dtype=np.complex128)

    def rho(self, c0):
        '''
        density function.
        c0: configuration list
        returns: rho(c)
        '''
        # c = relabel(c0)
        c = np.array(c0)
        ### needs to use +2,1,-1,-2. but everything else needs 0-3...
        return mcy.rho_cy(self.a1, self.a2, self.a3, self.b, self.u, self.v, self.w, c)


    def params(self):
        '''
        returns an array of parameters.

        '''
        p = np.zeros(self.Np,dtype=np.complex128)

        p[:self.Nv] = self.a1
        p[self.Nv:2*self.Nv] = self.a2
        p[2*self.Nv:3*self.Nv] = self.a3

        p[3*self.Nv:3*self.Nv + self.Nh] = self.b

        p[self.Nv*3 + self.Nh : self.Nv*3 + self.Nh + self.Nh*self.Nv] = np.reshape(self.u,self.Nh*self.Nv)
        p[self.Nv*3 + self.Nh + self.Nh*self.Nv:self.Nv*3 + self.Nh + 2*self.Nh*self.Nv] = np.reshape(self.v,self.Nh*self.Nv)
        p[self.Nv*3 + self.Nh + 2*self.Nh*self.Nv:] = np.reshape(self.w,self.Nh*self.Nv)

        return p

    def set_rand(self, O=0.0001, seed = None):
        '''
        randomly initialises parameters
        '''
        rng = np.random.default_rng(seed)

        self.a1 = O * ( rng.random(self.Nv)           + 1j * rng.random(self.Nv) )
        self.a2 = O * ( rng.random(self.Nv)           + 1j * rng.random(self.Nv) )
        self.a3 = O * ( rng.random(self.Nv)           + 1j * rng.random(self.Nv) )

        self.b =  O * ( rng.random(self.Nh)           + 1j * rng.random(self.Nh) )

        self.u =  O * ( rng.random((self.Nh,self.Nv)) + 1j * rng.random((self.Nh,self.Nv)) )
        self.v =  O * ( rng.random((self.Nh,self.Nv)) + 1j * rng.random((self.Nh,self.Nv)) )
        self.w =  O * ( rng.random((self.Nh,self.Nv)) + 1j * rng.random((self.Nh,self.Nv)) )

    def set_param(self,p):
        '''
        set params to specific values.
        p: vector of all parameters.
        '''
        self.a1 = p[0:self.Nv]
        self.a2 = p[self.Nv:self.Nv*2]
        self.a3 = p[self.Nv*2:self.Nv*3]

        self.b = p[self.Nv*3:self.Nv*3 + self.Nh]

        self.u = np.reshape(p[self.Nv*3 + self.Nh:self.Nv*3 + self.Nh + self.Nh*self.Nv],(self.Nh,self.Nv))
        self.v = np.reshape(p[self.Nv*3 + self.Nh + self.Nh*self.Nv:self.Nv*3 + self.Nh + 2*self.Nh*self.Nv],(self.Nh,self.Nv))
        self.w = np.reshape(p[self.Nv*3 + self.Nh + 2*self.Nh*self.Nv:],(self.Nh,self.Nv))

    def log_deriv(self, c):
        '''
        calculates derivatives at configuration c
        '''
        return mcy.log_deriv_cy(self.Nv,self.Nh,self.b,self.u,self.v,self.w,c)



        # return log_deriv_cy(self.Nv,self.Nh,self.b,self.u,self.v,self.w,c)

    def update(self,updates):
        '''
        updates: vector of parameter updates
        '''
        self.a1 += updates[0 : self.Nv]
        self.a2 += updates[self.Nv : self.Nv*2]
        self.a3 += updates[self.Nv*2 : self.Nv*3]
        #
        self.b += updates[self.Nv*3 : self.Nv*3+self.Nh]

        '''
        # does the reshaping lead to the problems when N!=M?
        # '''
        self.u += np.reshape(updates[self.Nv*3+self.Nh : self.Nv*3+self.Nh+self.Nv*self.Nh],(self.Nh,self.Nv))
        self.v += np.reshape(updates[self.Nv*3+self.Nh+self.Nv*self.Nh : self.Nv*3+self.Nh+self.Nv*self.Nh*2],(self.Nh,self.Nv))
        self.w += np.reshape(updates[self.Nv*3+self.Nh+self.Nv*self.Nh*2:],(self.Nh,self.Nv))



#end
