import numpy as N
from scipy.stats import norm
import hankel3d as hank
import itertools
import warnings
import matplotlib.pyplot as plt
import time
import pathos.pools as pp
import os

class ITAM:
    """
    To run itam see notebook for examples.

    Input::
     boxsize: of the simulation
     ng: grid resolution (per side)
     Rth: smoothing scale
     nmax: for the Wiener-Khinchin transform
     stepzise: for the Wiener-Khinchin transform
     beta: for the update of the pre-translation power spectrum
     eps: convergence paraeter
     pathto_pk: lookup table with target simulation
     pathto_ppf: lookup table with target PDF
     pathto_rescale: rescaling factor of the power spectrum
     saveto: folder where to save
     plotty: boolean to show the convergence to target power spectrum

    Methods::
     The parameters of each method are in their docs
      realization_g: make a realization of a gaussian field on the grid
      realization_ng: translate a gaussian realization to the target
      realPower:
      make_covariance: make many realizations
      fastPk: measure the power spectrum of a realization

    Output::
      the class stores the pre-translation power spectrum pk_g and the translated one pk_ng, 
      to be compared to the target, also stored as pk. 
    """

    def __init__( self , boxsize=256., ng=256 , Rth = 2. ,  nmax=10000 , stepsize=1e-04 , beta=1.0 , eps = 0.001, Deps=0.001, plotty=0,
            pathto_pk=None,pathto_ppf=None, pathto_rescale=None,saveto=None):

        self.nmax = nmax            # for hankel transform
        self.stepsize = stepsize    # for hankel transform

        self.beta = beta            # update pk
        self.eps = eps
        self.Deps = Deps

        self.boxsize = boxsize
        self.ng = ng

        self.Rth = Rth

        self.plotty = plotty

        self.saveto = saveto

        if not os.path.exists(pathto_pk):
            raise ValueError("Select correctly the path to lookup table of target power spectrum")
        else:
            kbins,pk = N.loadtxt(pathto_pk)

        if not os.path.exists(pathto_ppf):
            raise ValueError("Select correctly the path to lookup table of target percent point function")
        else:
            self.cdf,self.log_ppf = N.loadtxt(pathto_ppf)
            self.mini=self.log_ppf[0]
            self.maxi=self.log_ppf[-1]

        if not os.path.exists(pathto_rescale):
            raise ValueError("Select correctly the path to rescaling factor of power spectrum")
        else:
            correction = N.loadtxt(pathto_rescale)

        if not os.path.exists(saveto):
            raise ValueError("create a folder where to save the results")

        pk *= correction
        cellsize = boxsize/float(ng)
        lmin = boxsize/float(ng)/10.
        lmax = boxsize
        self.kmin = 2.*N.pi/lmax
        self.kmax = 2.*N.pi/lmin
        self.k = N.logspace( N.log10(self.kmin) , N.log10(self.kmax) , 200 )
        self.r = N.logspace( N.log10(lmin) , N.log10(lmax) , 200 )
        self.pk = 10.**N.interp(N.log10(self.k),N.log10(kbins),N.log10(pk),left=0.,right=0.)
        Wk2 = N.exp(-self.k*self.k*Rth*Rth)
        self.pk *= Wk2

        self.pk_g , self.pk_ng = self.itam()

        k_pk_g = N.vstack((self.k,self.pk_g))
        k_pk_ng = N.vstack((self.k,self.pk_ng))

        N.savetxt(saveto+'pk_itam_linear_table.txt',k_pk_g)
        N.savetxt(saveto+'pk_itam_table.txt',k_pk_ng)

        if plotty==1:
            plt.figure(figsize=(1.62*5.5,5.5))
            with warnings.catch_warnings():
                warnings.simplefilter( "ignore" , category = RuntimeWarning )
                plt.semilogx(self.k,(self.pk-self.pk_ng)/self.pk,'--',lw=2.)
            plt.ylim([-0.01,0.01])
            plt.xlim([0.02,1.5])
            plt.xlabel('$k \ [h/Mpc]$',fontsize='xx-large')
            plt.ylabel('$\Delta P(k) \ [Mpc/h]^3$',fontsize='xx-large')
            plt.show()


    def itam(self):
        ''' the main algorithm '''

        target_s = self.pk
        s_g_iterate = self.pk

        eps0 = 1.
        eps1 = 1.
        ii = 0

        Deps = 1.

        while Deps > self.Deps and eps1>self.eps:
            ii += 1
            print('iteration =', ii)
            eps0 = eps1

            r_g_iterate = hank.pk_to_xi( s_g_iterate , self.r , self.k, self.nmax, self.stepsize )
            if N.any( N.isnan(r_g_iterate)):
                raise ValueError("r_g_iterate contains NaN")

            sigma2 = r_g_iterate[0]
            r_ng_iterate = N.asarray([ self.solve_integral(r_g/sigma2,sigma2) for r_g in r_g_iterate ])
            
            if N.any( N.isnan(r_ng_iterate) ):
                raise ValueError("r_ng_iterate contains NaN")

            s_ng_iterate = hank.xi_to_pk( r_ng_iterate , self.r , self.k, self.nmax, self.stepsize*10 )

            eps1 = N.sqrt( N.sum((target_s - s_ng_iterate) ** 2. )/N.sum(target_s**2) )
            Deps = abs(eps1-eps0)
            Deps/= eps1
            print('eps = %.5f' % eps1,'Deps =%.5f' % Deps)

            if Deps > self.Deps and eps1>self.eps:

                with warnings.catch_warnings():
                    warnings.simplefilter( "ignore" , category = RuntimeWarning )

                    s_g_iterate = N.power( target_s / s_ng_iterate, self.beta ) * s_g_iterate
                    s_g_iterate[N.isnan(s_g_iterate)] = 0.

                if N.any( N.isnan(s_g_iterate) ):
                    raise ValueError("s_g_iterate contains NaN")

            else:
                print('converged at', ii, 'iteration')

                return s_g_iterate,s_ng_iterate


    def solve_integral(self,rho,sigma2):
        ''' Gauss-Hermite quadrature like in
        https://gist.github.com/markvdw/f9ca12c99484cf2a881e84cb515b86c8 '''

        if rho >= 1.0:
            rho = 1-1e-08
 
        nhg = 30
        x, w = N.polynomial.hermite.hermgauss(nhg)
        Sigma = sigma2*N.array([[1., rho], [rho, 1.]])
        Nd = 2
        const = N.pi**(-0.5*Nd)
        
        # gaussian variable
        xn = N.array(list(itertools.product(*(x,)*Nd)))

        # gauss hermite weights
        wn = N.prod(N.array(list(itertools.product(*(w,)*Nd))), 1)

        # normalized diagonal variables
        yn = 2.0**0.5*N.dot(N.linalg.cholesky(Sigma), xn.T).T

        yn = norm.cdf( yn ,loc=0., scale= N.sqrt(sigma2) )
                 
        gn = N.power( 10. , N.interp(yn, self.cdf , self.log_ppf ,left=self.mini,right=self.maxi) )-1.

        gn = N.prod( gn, 1 )

        if not N.all( N.isfinite( gn ) ):
            gn[N.where(N.isinf(gn))] = 0.

        z =  N.sum( (wn * const ) * gn , axis=0 )

        return z


    def getkgrid(self):
        ''' It returns a grid of k in fft format '''

        kmin = 2*N.pi/N.float(self.boxsize)
        sh = (self.ng,self.ng,self.ng/2+1)
        kx,ky,kz = N.mgrid[0:sh[0],0:sh[1],0:sh[2]].astype(N.float64)

        kx[N.where(kx > self.ng/2)] -= self.ng
        ky[N.where(ky > self.ng/2)] -= self.ng
        kz[N.where(kz > self.ng/2)] -= self.ng

        kx *= kmin
        ky *= kmin
        kz *= kmin

        k = N.sqrt(kx**2+ky**2+kz**2)

        return k


    def realization_g(self,seed=1):
        '''
        It makes a realization with the pre-tranlation power spectrum.
        Inputs::
            seed: seed of random number generator
        Outputs::
            the fourier space random field
        '''

        r = N.random.RandomState(seed)
        kgrid = self.getkgrid()
        shc = N.shape(kgrid)
        sh = N.prod(shc)

        dk = N.empty(sh,dtype=N.complex64)
        dk.real = r.normal(size=sh).astype(N.float32)
        dk.imag = r.normal(size=sh).astype(N.float32)
        dk /= N.sqrt(2.)

        with warnings.catch_warnings():
            warnings.simplefilter( "ignore" , category = RuntimeWarning )
            pk = N.power(10.,N.interp(N.log10(kgrid.flatten()),N.log10(self.k),N.log10(self.pk_g),right=0)).astype(N.complex64)

        pk[ pk < 0. ] = 0.
        pk[ N.isnan(pk) ] = 0.

        dk *= N.sqrt(pk)/self.boxsize**1.5 * self.ng**3.
        dk[0] = 0.
        dk = N.reshape(dk,shc)

        # Hermitian symmetric: dk(-k) = conjugate(dk(k))
        dk[self.ng // 2 + 1:, 1:,
            0] = N.conj(N.fliplr(N.flipud(dk[1:self.ng // 2, 1:, 0])))
        dk[self.ng // 2 + 1:, 0, 0] = N.conj(dk[self.ng // 2 - 1:0:-1, 0, 0])
        dk[0, self.ng // 2 + 1:, 0] = N.conj(dk[0, self.ng // 2 - 1:0:-1, 0])
        dk[self.ng // 2, self.ng // 2 + 1:,
            0] = N.conj(dk[self.ng // 2, self.ng // 2 - 1:0:-1, 0])

        dk[self.ng // 2 + 1:, 1:, self.ng //
            2] = N.conj(N.fliplr(N.flipud(dk[1:self.ng // 2, 1:, self.ng // 2])))
        dk[self.ng // 2 + 1:, 0, self.ng //
            2] = N.conj(dk[self.ng // 2 - 1:0:-1, 0, self.ng // 2])
        dk[0, self.ng // 2 + 1:, self.ng //
            2] = N.conj(dk[0, self.ng // 2 - 1:0:-1, self.ng // 2])
        dk[self.ng // 2, self.ng // 2 + 1:, self.ng //
            2] = N.conj(dk[self.ng // 2, self.ng // 2 - 1:0:-1, self.ng // 2])

        return dk


    def realization_ng(self,seed=1):
        '''
        It applies the translation process transform to a gaussian field realization.
        Inputs are the same as in *realization_g* method.
        Outputs::
            gaussian field and nonlinear field
        '''

        dk = self.realization_g(seed)
        d_g = N.fft.irfftn(dk)
        d_ng = N.power( 10. , N.interp( norm.cdf( d_g, loc = 0. , scale = N.std(d_g) ) ,
        self.cdf , self.log_ppf ,left=self.mini,right=self.maxi) ) -1.

        return d_g,d_ng


    def realPower(self,seed):
        ''' Function to parallelize the computation of covariance. '''

        d_g, d_ng = self.realization_ng(seed)
        dk_g = N.fft.rfftn(d_g)
        dk_ng = N.fft.rfftn(d_ng)
        kgrid = self.getkgrid().flatten()
        ps_g = self.fastPk( dk_g , kgrid )
        ps_ng = self.fastPk( dk_ng , kgrid )
        
        return ps_g, ps_ng


    def make_covariance(self,nreal=10,parallel=1,cores=2):
        '''
        To make a test for the covariance matrix of power spectrum.
        Inputs::
          nreal: number of realizations
        Outputs::
          saves the power spectra and the kbins on files.
        '''
        # seeds
        seeds = N.arange(int(nreal))
        print('making', nreal, 'realizations')

        # bins in k space
        nkbins = self.ng*3//4
        kgrid = self.getkgrid().flatten()
        knz = kgrid[kgrid>0.]
        delta = (self.ng*N.pi/self.boxsize - 2*N.pi/self.boxsize)/(nkbins+1)/2.
        kbins = N.linspace(2*N.pi/self.boxsize-delta,self.ng*N.pi/self.boxsize-delta,nkbins+1)
        counts = N.histogram(N.log10(knz),N.log10(kbins),range=(2*N.pi/self.boxsize-delta,self.ng*N.pi/self.boxsize-delta))[0]
        kbins = kbins[0:nkbins]+N.diff(kbins)/2.
        kbins = kbins[counts>0]
        print('saving data_itam/kbins.txt')
        N.savetxt(self.saveto+'kbins.txt',kbins)

        # power spectra
        t0 = time.time()

        if parallel==1:
            print('implementing parallel execution with', cores, 'cores')
            pool = pp.ProcessPool(cores)
            ppp = pool.map( self.realPower, range(nreal) )
            psvals_g = N.asarray( [ ppp[i][0] for i in range(nreal)] )
            psvals_ng = N.asarray( [ ppp[i][1] for i in range(nreal)] )

        else:
            print('serial execution')
            psvals_g = N.zeros(len(kbins))
            psvals_ng = N.zeros(len(kbins))
            variances = []
            for j in range(nreal):
                d_g,d_ng = self.realization_ng(j)
                variances = N.append(variances, N.var(d_ng))
                dk_g = N.fft.rfftn(d_g)
                dk_ng = N.fft.rfftn(d_ng)
                psiter =  self.fastPk( dk_g , kgrid )
                psvals_g = N.vstack((psvals_g,psiter,))
                psiter =  self.fastPk( dk_ng , kgrid )
                psvals_ng = N.vstack((psvals_ng,psiter,))

            psvals_g = psvals_g[1::,:]
            psvals_ng = psvals_ng[1::,:]

        t1 = time.time()
        total=(t1-t0)/60.
        print('spent', total, 'minutes')
        print( total/float(nreal) , 'minutes per realization')

        print('saving data_itam/psvals_g.txt')
        print('saving data_itam/psvals_ng.txt')

        N.savetxt(self.saveto+'psvals_g.txt',psvals_g)
        N.savetxt(self.saveto+'psvals_ng.txt',psvals_ng)

        return 0


    def fastPk(self,dk,kgrid):
        '''
        Compute Pk for a field on the grid.
        Inputs::
            dk: fft of field.
            kgrid: k-grid in fft format
        '''
        kgrid = kgrid.flatten()
        nkbins = self.ng*3//4
        dk2 = abs( dk.flatten() ) ** 2.
        dk2 = dk2[kgrid>0.]
        knz = kgrid[kgrid>0.]

        delta =(self.ng*N.pi/self.boxsize - 2*N.pi/self.boxsize)/(nkbins+1)/2.
        kbin = N.linspace(2*N.pi/self.boxsize-delta,self.ng*N.pi/self.boxsize-delta,nkbins+1)
        ps = N.histogram(N.log10(knz),N.log10(kbin),weights=dk2,range=(2*N.pi/self.boxsize-delta,self.ng*N.pi/self.boxsize-delta))[0]
        counts = N.histogram(N.log10(knz),N.log10(kbin),range=(2*N.pi/self.boxsize-delta,self.ng*N.pi/self.boxsize-delta))[0]
        binvals = kbin[0:nkbins]+N.diff(kbin)/2.
        binvals = binvals[counts>0]
        ps = ps[counts>0]
        counts = counts[counts>0]
        ps = ps/counts

        norm = self.boxsize**3/self.ng**6

        return ps * norm


if __name__== "__main__":
     itm = ITAM(boxsize=256.,ng=256, beta=0.5,Rth=1.,eps=0.05,Deps=0.001,
            pathto_pk='data_itam/planck_pk.txt',pathto_ppf = 'data_itam/_ppf.txt',saveto='data_itam/',
            pathto_rescale='data_itam/rescale_factor.txt' , plotty=1)

     itm.make_covariance(nreal=3,parallel=0,cores=2)
     #itm.make_covariance(nreal=3,parallel=1,cores=2)
