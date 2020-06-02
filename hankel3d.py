import numpy as N
import bessint as BI

class Hankel3D:
    def __init__(self,nMax):
        """

        """
        self.bessint = BI.BesselIntegrals(0.5,nMax)

    def transform1(self,f,x,n,h,pk2xi=1):
        """

        """
        bi = 1.0/x**3*self.bessint.besselInt(lambda z:z**1.5*f(z/x),n,h)
        pf = 1.0/(2.0*N.pi)**1.5
        if pk2xi==0:
            pf =1.0/pf
        return pf*bi

    def transform(self,f,x,n,h,pk2xi=1):
        """

        """
        if N.isscalar(x):
            return self.transform1(f,x,n,h,pk2xi)
        else:
            return N.array(list(map(lambda z:self.transform1(f,z,n,h,pk2xi),x)))


def interpolateLin(y,x,xNew):
    """
    linear interpolation of y[x] onto y[xNew]
    Linearly extrapolates if outside range
    """
    xInd = N.clip(N.searchsorted(x,xNew)-1,0,len(x)-2)
    xFract = (xNew-x[xInd])/(x[xInd+1]-x[xInd])
    return y[xInd]+xFract*(y[xInd+1]-y[xInd])


def pk_to_xi(pk,r,k,nmax,stepsize):

    hank = Hankel3D(nmax)
    xi = hank.transform( lambda x: interpolateLin(pk,k,x), r , nmax , stepsize )
    return xi


def xi_to_pk(xi,r,k,nmax,stepsize):

    hank = Hankel3D(nmax)
    pk = hank.transform(lambda x: interpolateLin(xi,r,x),k,nmax,stepsize,pk2xi=0)
    return pk
