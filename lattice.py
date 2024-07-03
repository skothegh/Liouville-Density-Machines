import numpy as np

class Base_Lattice():
    """
    Base Lattice class which other lattice classes should (?) inherit from.

    Length (int): length of one side of the lattice
    ndim (int): number of dimensions
    pbc (bool): periodic boundary conditions

    nsites (int): total number of sites
    sites :
    nbonds (int): number of pairwise connections between sites

    """
    def __init__(self,length,ndim=1,pbc=False):
        self.length = length
        self.ndim = ndim
        self.pbc = pbc

        self.nsites = None

        self.bonds = None
        self.sites = None
        self.nbonds = None


    def __eq__(self,x):
        """
        compares lattice with other lattice
        self.eq(x) -> bool

        x: lattice
        return (bool)
        """
        eq = True

        eq *= x.type == self.type
        eq *=  (x.length == self.length)
        eq *=  (x.ndim == self.ndim)
        eq *=  (x.pbc is self.pbc)
        eq *=  (x.nsites == self.nsites)

        return bool(eq)

class Triangular_Lattice():
    """
    Triangular Lattice class

    Length (int): length of one side of the lattice
    ndim (int): number of dimensions
    pbc (bool): periodic boundary conditions

    nsites (int): total number of sites
    sites [int]: array of ints, which enumerate the sites
    nbonds (int): number of pairwise connections between sites

    sadly, only supports ndim <=2  so far

    EXAMPLE:
    ndim = 1
    len = 2
    pbc = false

      o---o
     / \ /
    o---o

    """
    def __init__(self,length,ndim=1,pbc=False):
        self.length = length
        self.ndim = ndim
        self.pbc = pbc

        self.nsites = None

        self.bonds = None
        self.sites = None
        self.nbonds = None

        self.type = "Tringular"

        self.build_graph()

    def __eq__(self,x):
        eq = True

        # eq *= isinstance(x,Triangular_Lattice)
        eq *= x.type == self.type
        eq *=  (x.length == self.length)
        eq *=  (x.ndim == self.ndim)
        eq *=  (x.pbc is self.pbc)
        eq *=  (x.nsites == self.nsites)

        return bool(eq)

    def build_graph(self):
        '''
        takes length, ndim, pbc to create a list of bonds and site indices.

        initializes self.bonds, self.sites, self.nsites, self.nbonds

        '''


        d = self.ndim
        pbc = int(self.pbc)
        l = self.length

        if(d == 1):
            A = np.arange(l * 2)                                # list of sites; n=l*2, since the 1d tri lattice is effectively a ladder, see above

            Bh = np.zeros(((l-1+pbc)*2 ,2),dtype=np.int32)      # horizontal bonds
            Bb = np.zeros((l,2),dtype=np.int32)                 # "forward" bonds \
            Bf = np.zeros((l-1+pbc,2),dtype=np.int32)           # "backward" bonds /

            for i in range(Bh.shape[0]):
                Bh[i,0] = i
                Bh[i,1] = (i + 2) % (2*l)

            for i in range(Bb.shape[0]):
                Bb[i,0] = (i*2)
                Bb[i,1] = (i*2) + 1

            for i in range(Bf.shape[0]):
                Bf[i,0] = (i*2)
                Bf[i,1] = ((i*2) + 3) % (2*l)

            # create one array of arrays which contains all bonds
            B = np.vstack(
                          [
                           Bh,
                           Bb,
                           Bf
                          ]
                         )
        # same as above except for 2d
        elif(d == 2):

            A = np.arange(l ** 2).reshape(l,l).T

            Bh = np.zeros((l,l-1+pbc,2),dtype=np.int32)
            Bb = np.zeros((l-1+pbc,l,2),dtype=np.int32)
            Bf = np.zeros((l-1+pbc,l-1+pbc,2),dtype=np.int32)

            for i in range(Bh.shape[0]):
                for j in range(Bh.shape[1]):
                    Bh[i,j,0] = A[i,j]
                    Bh[i,j,1] = A[i,(j+1) % l]

            for i in range(Bb.shape[0]):
                for j in range(Bb.shape[1]):
                    Bb[i,j,0] = A[i,j]
                    Bb[i,j,1] = A[(i+1) % l,j]

            for i in range(Bf.shape[0]):
                for j in range(Bf.shape[1]):
                    Bf[i,j,0] = A[i,j]
                    Bf[i,j,1] = A[(i+1) % l, (j+1) % l]

            B = np.vstack(
                          [
                           Bh.reshape(Bh.shape[0]*Bh.shape[1],2),
                           Bb.reshape(Bb.shape[0]*Bb.shape[1],2),
                           Bf.reshape(Bf.shape[0]*Bf.shape[1],2)
                          ]
                         )

        self.sites = A
        self.nsites = A.size
        self.bonds = B
        self.nbonds = len(B)


class Linear_Lattice():
        """
        Linear Lattice class. 1d = chain, 2d = grid, 3d = cube

        Length (int): length of one side of the lattice
        ndim (int): number of dimensions
        pbc (bool): periodic boundary conditions

        nsites (int): total number of sites
        sites [int]: array of ints, which enumerate the sites
        nbonds (int): number of pairwise connections between sites

        sadly, only supports ndim <=2  so far

        EXAMPLE:
        ndim = 2
        len = 2
        pbc = false

        o---o
        |   |
        o---o

        """
    def __init__(self,length,ndim=1,pbc=False):
        self.length = length
        self.ndim = ndim
        self.pbc = pbc

        self.nsites = length**ndim

        self.bonds = None
        self.sites = None
        self.nbonds = None

        self.type = "Cubic"

        self.build_graph()


    def __eq__(self,x):
        eq = True

        eq *= isinstance(x,Lattice)
        eq *=  (x.length == self.length)
        eq *=  (x.ndim == self.ndim)
        eq *=  (x.pbc is self.pbc)
        eq *=  (x.nsites == self.nsites)

        return bool(eq)

    def build_graph(self):
        '''
        Sites are numbered top-left to bottom-right
        Requires nearest neighbours
        Only cube-like lattices (chains, squares, cubes...) and only 2d so far
        '''
        assert self.ndim < 3, "not tested for cubes yet..."

        d = self.ndim
        pbc = int(self.pbc)
        l = self.length

        if(d == 1):
            A = np.arange(l)

            B = np.zeros((l-1+pbc,2),dtype=np.int32)

            for i in range(B.shape[0]):
                B[i,0] = i
                B[i,1] = (i + 1) % l

        elif(d == 2):
            nb = 2*l*(l-1 + pbc)

            A = np.arange(l**2).reshape(l,l)

            B1 = np.zeros((l-1+pbc,l,2),dtype=np.int32)
            B2 = np.zeros((l,l-1+pbc,2),dtype=np.int32)

            for i in range(B1.shape[0]):
                for j in range(B1.shape[1]):
                    B1[i,j,0] = A[i,j]
                    B1[i,j,1] = A[(i+1) % l,j]

            for i in range(B2.shape[0]):
                for j in range(B2.shape[1]):
                    B2[i,j,0] = A[i,j]
                    B2[i,j,1] = A[i,(j+1) % l]

            B = np.vstack(
                          [
                           B1.reshape(B1.shape[0]*B1.shape[1],2),
                           B2.reshape(B2.shape[0]*B2.shape[1],2)
                          ]
                         )

        self.sites = A
        self.bonds = B
        self.nbonds = len(B)
