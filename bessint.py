"""
   ``Gaussian'' quadrature of integrals mulitplied with
   bessel functions.
   
  (c) 2004, Istvan Szapudi, Ifa, Hawaii

  Thu Dec 16 15:02:57 HST 2004

  Modified, Adrian Pope, IfA, Hawaii
  Fri Nov 24 00:37:17 HST 2006

  Modified, Federico Tosone,
  Mon May 18 11:50:00 GMT+2 2020
"""

import scipy.special as SS
import numpy as N

class BesselIntegrals:
    """
      Numerical Integration of Bessel Functions based
      on Ogata (2004)
    """

    def __init__(self,nu,nMax):
        """
        initialization, calculates zeros of the Bessel functions
        and weights.
        
        nu: order of the Bessel function
        nMax: maximum number of terms to use
        """
        self.nu = nu
        self.nMax = nMax
        self.z = self.zeros(nu,nMax)
        self.w = self.weights(nu,self.z)


    def phi1(self,t):
        return N.fabs(t)

    def phi2(self,t):
        return t*N.tanh(0.5*N.pi*N.sinh(t))
    
    def phi(self,t):
        """
        double exponential transformation
        """
        if N.isscalar(t):
            t = N.array([t])
        return N.piecewise(t,[N.fabs(t)>5.0],[self.phi1,self.phi2])


    def phiPrime1(self,t):
        return N.sign(t)

    def phiPrime2(self,t):
        return 0.5*N.pi*t*N.cosh(t)/N.cosh(0.5*N.pi*N.sinh(t))**2+N.tanh(0.5*N.pi*N.sinh(t))
        
    def phiPrime(self,t):
        """
        derivative of the double exponential transform
        """
        if N.isscalar(t):
            t = N.array([t])
        return N.piecewise(t,[N.fabs(t)>5.0],[self.phiPrime1,self.phiPrime2])


    def zeros(self,nu,nMax):
        """
        returns the first nMax zeros of the Bessel function of order nu.
        

        """
        if nu==0.5:
            return N.arange(1.0,nMax+1)
        else:
            return N.array(map(lambda x: SS.jn_zeros(nu,x)[0]/N.pi,N.arange(1,nMax+1)))


    def weights(self,nu,zeroArr):
        """
        return the weights for quadrature
        w_{ mu k} = Y_mu( pi xi_{nu k}/J_nu+1(Pi xi_{nu k})
        """
        piXi = N.pi*zeroArr
        return SS.yv(nu,piXi)/SS.jv(nu+1,piXi)


    def besselInt(self,f,nMax,h):
        """
        ``Gauss''-type quadrature for bessel functions realizing
        int f(x) J_nu(x)

        h is the step
        """
        if nMax > self.nMax:
            print('Error: nMax = '+str(nMax)+' > self.nMax = '+str(self.nMax))
            return 0.0
        hXi = h*self.z[0:nMax]
        w = self.w[0:nMax]
        pih = N.pi/h
        phiXi = pih*self.phi(hXi)
        a = N.pi*w*f(phiXi)*SS.jv(self.nu,phiXi)*self.phiPrime(hXi)
        return N.sum(a)
