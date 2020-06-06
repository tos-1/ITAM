import numpy as N
from scipy.stats import norm,lognorm
import hankel3d as hank
import itertools
import warnings
import matplotlib.pyplot as plt

class logITAM:
    """
    Same as ITAM, but assuming lognormal PDF
    Input::
     boxsize: of the simulation
     ng: grid resolution (per side)
     Rth: smoothing scale
     nmax: for the Wiener-Khinchin transform
     stepzise: for the Wiener-Khinchin transform
     beta: for the update of the pre-translation power spectrum
     eps: convergence paraeter
     pathto_pk: lookup table with target simulation
     plotty: boolean to show the convergence to target power spectrum

    Output::
      the class stores the pre-translation power spectrum pk_g and the translated one pk_ng, 
      to be compared to the target, also stored as pk. 
    """

    def __init__( self , boxsize=256., ng=256 , Rth = 2. ,  nmax=10000 , stepsize=1e-04 , beta=1.0 , eps = 0.001, Deps=0.001, plotty=0,
            pathto_linpk=None, pathto_pk=None):

        self.nmax = nmax            # for hankel transform
        self.stepsize = stepsize    # for hankel transform

        self.beta = beta            # update pk
        self.eps = eps
        self.Deps = Deps

        self.boxsize = boxsize
        self.ng = ng

        self.Rth = Rth

        self.plotty = plotty

        try:
            kbins,pk = N.loadtxt(pathto_pk)
        except:
            raise ValueError("Select correctly the path to lookup table of target power spectrum")

        if not pathto_linpk==None:
            self.flag_lin = True
            print('you specified the linear power spectrum for the initilization')
            if not os.path.exists(pathto_linpk):
                raise ValueError("The path to the linear power spectrum does not exist")
            else:
                kbins,pk_l = N.loadtxt(pathto_pk)
        else:
            self.flag_lin = False
            pass

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

        if self.flag_lin == True:
            pk_l *= correction
            self.pk_l = 10.**N.interp(N.log10(self.k),N.log10(kbins),N.log10(pk_l),left=0.,right=0.)
            Wk2 = N.exp(-self.k*self.k*Rth*Rth)
            self.pk_l *= Wk2

        self.pk_g , self.pk_ng = self.itam()

        # Coles and Jones case
        xi_ng = hank.pk_to_xi( self.pk , self.r , self.k , self.nmax , self.stepsize )
        xi_g = N.log1p(xi_ng)
        self.pk_g_CJ = hank.xi_to_pk( xi_g , self.r , self.k , self.nmax , self.stepsize*10 )

        ## Eq.20 case
        #xi_g = hank.pk_to_xi( self.pk_g , self.r , self.k , self.nmax , self.stepsize )
        #xi_ng = N.exp(xi_g[0])*(N.exp(xi_g)-1.)
        #self.pk_ng_exact = hank.xi_to_pk( xi_ng , self.r , self.k , self.nmax , self.stepsize*10 )


        #self.test_realization(seed=31415)

        if plotty==1:
            plt.figure(figsize=(1.62*5.5,5.5))
            with warnings.catch_warnings():
                warnings.simplefilter( "ignore" , category = RuntimeWarning )
                plt.semilogx(self.k,(self.pk_g_CJ-self.pk_g)/self.pk_g_CJ,'-',lw=2.,label='pre-translation')
                plt.semilogx(self.k,(self.pk-self.pk_ng_exact)/self.pk,'--',lw=2.,label='itam')
            plt.grid()
            plt.legend
            plt.ylim([-0.05,0.05])
            plt.xlim([0.02,1.5])
            plt.xlabel('$k \ [h/Mpc]$',fontsize='xx-large')
            plt.ylabel('$\Delta P(k) \ [Mpc/h]^3$',fontsize='xx-large')
            plt.show()


    def itam(self):
        ''' the main algorithm '''

        target_s = self.pk

        if self.flag_lin == True:
            s_g_iterate = self.pk_l
        else:
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

            if N.any( N.isnan(r_g_iterate) ):
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

        gn = lognorm.ppf( yn , s = N.sqrt(sigma2) ,loc=0.0, scale=1.0)

        # Eq. 16
        gn *= N.exp(-sigma2/2.)
        gn -= 1.

        # Eq. 20
        #gn -= N.exp(sigma2/2.)-1.

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


    def test_realization(self,seed=1):
        '''
        To make a realization with the newly found power spectrum.
        Inputs::
            seed: seed of random number generator
            amp: whether to vary the amplitudes of the power spectrum. If 0 (default) they are fixed.
        Outputs::
            the fourier space random field simulation
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
        dk[self.ng/2+1:,1:,0]= N.conj(N.fliplr(N.flipud(dk[1:self.ng/2,1:,0])))
        dk[self.ng/2+1:,0,0] = N.conj(dk[self.ng/2-1:0:-1,0,0])
        dk[0,self.ng/2+1:,0] = N.conj(dk[0,self.ng/2-1:0:-1,0])
        dk[self.ng/2,self.ng/2+1:,0] = N.conj(dk[self.ng/2,self.ng/2-1:0:-1,0])

        dk[self.ng/2+1:,1:,self.ng/2]= N.conj(N.fliplr(N.flipud(dk[1:self.ng/2,1:,self.ng/2])))
        dk[self.ng/2+1:,0,self.ng/2] = N.conj(dk[self.ng/2-1:0:-1,0,self.ng/2])
        dk[0,self.ng/2+1:,self.ng/2] = N.conj(dk[0,self.ng/2-1:0:-1,self.ng/2])
        dk[self.ng/2,self.ng/2+1:,self.ng/2] = N.conj(dk[self.ng/2,self.ng/2-1:0:-1,self.ng/2])

        d_g = N.fft.irfftn(dk)
        print('mean of Gaussian field=',N.mean(d_g,axis=None))
        cdf = norm.cdf( d_g ,loc= N.mean(d_g), scale= N.std(d_g) )
        d_ng = lognorm.ppf( cdf , s = N.std(d_g) ,loc=0.0, scale=1.0)

        # Eq. 16
        d_ng *= N.exp(-N.var(d_g)/2.)
        d_ng -= 1.

        # Eq. 20
        #d_ng -= N.exp(N.var(d_g)/2.)

        return d_ng
