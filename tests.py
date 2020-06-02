import numpy as N
import hankel3d
from scipy.integrate import quad
from scipy.stats import norm, lognorm
import itertools
import matplotlib.pyplot as plt


def test_integral(rho=0.7,sigma2=1):
    ''' Test the Gauss-Hermite integration,
    https://gist.github.com/markvdw/f9ca12c99484cf2a881e84cb515b86c8
    '''

    if rho >= 1.0:
        rho = 0.9999

    print('rho=',rho)
    print('sigma^2=',sigma2)
    nhg = 30
    x, w = N.polynomial.hermite.hermgauss(nhg)
    Sigma = sigma2 * N.array([[1., rho], [rho, 1.]])
    Nd = 2
    const = N.pi**(-0.5*Nd)

    # gaussian variable
    xn = N.array(list(itertools.product(*(x,)*Nd)))
    
    # gauss hermite weights
    wn = N.prod(N.array(list(itertools.product(*(w,)*Nd))), 1)

    # normalized diagonal variables
    yn = 2.0**0.5*N.dot(N.linalg.cholesky(Sigma), xn.T).T

    ## basic tests
    print("Normalising constant: %f" % N.sum(wn * const))
    print("Mean:")
    print(N.sum((wn * const)[:, None] * yn, 0))
    print("Covariance:")
    covfunc = lambda x: N.outer(x, x)
    print(N.sum((wn * const)[:, None, None] * N.array(list(map(covfunc, yn))), 0))

    return 0



def test_itam(nmax=10000,stepsize=1e-04,boxsize=256.,ng=256,target=None,Rth=2.):
    ''' check the identity case '''

    try:
        kbins,pk = N.loadtxt(target)
    except:
        ValueError("Select correctly the path to lookup table of target power spectrum")

    lmin = boxsize/float(ng)/10.
    lmax = boxsize
    kmin = 2.*N.pi/lmax
    kmax = 2.*N.pi/lmin

    k = N.logspace( N.log10(kmin) , N.log10(kmax) , 200 )    # spectral grid
    rr = N.logspace( N.log10(lmin) , N.log10(lmax) , 200 )   # physical grid
    kny = 2.0 * N.pi / boxsize * ng / 2.0

    kbins,pk = N.loadtxt('data_itam/planck_pk.txt')
    pk = 10.**N.interp(N.log10(k),N.log10(kbins),N.log10(pk),left=0.,right=0.)
    Wk2 = N.exp(-k*k*Rth*Rth)
    pk *= Wk2

    xi = hankel3d.pk_to_xi( pk , rr , k , nmax , stepsize )

    print('solving integral')
    cdf,ppf = N.loadtxt('data_itam/_ppf.txt')
    xi_ng = N.asarray([ solve_integral(rrho,cdf,ppf,xi[0]) for rrho in xi/xi[0] ])
    pkfromxi = hankel3d.xi_to_pk( xi_ng , rr , k , nmax , stepsize*10 )

    # correlation function
    plt.clf()
    plt.subplot(211)
    plt.xlim([-2.,150.])
    plt.plot(rr,rr**2*xi,label='before transform')
    plt.plot(rr,rr**2*xi_ng,'r--',label='after transform')
    plt.title('correlation')
    plt.legend()

    plt.subplot(212)
    plt.loglog(k,pkfromxi,lw=2,label='after translation')
    plt.loglog(k,pk,'r--',lw=2,label='initial')
    plt.xlabel(r'$k [h/{\rm Mpc}]$')
    plt.ylabel(r'$P(k)$')
    plt.legend()
    plt.xlim([kmin,kmax])
    plt.ylim([1.,1e+06])
    plt.show()

    def integrand(kk):
        ppk = N.interp(kk,k,pkfromxi,left=0.,right=0.)
        ppk *= kk*kk
        return ppk

    k_min = 2.0 * N.pi / boxsize
    k_max = 2.0 * N.pi / boxsize * ng / 2.

    sigma2, _ = quad(integrand, k_min, k_max, epsabs = 0.0, epsrel = 1e-03, limit = 100)
    sigma2 /= 2.0 * N.pi**2.
    print('sigma^2=',sigma2, 'whereas is expected=',xi_ng[0])

    return 0


def test_itam_lognormal(nmax=10000,stepsize=1e-04,boxsize=256.,ng=256,Rth=2.):
    '''
    Check ITAM gives the exact result in the lognormal case
    '''
    lmin = boxsize/float(ng)/10.
    lmax = boxsize
    kmin = 2.*N.pi/lmax
    kmax = 2.*N.pi/lmin

    k = N.logspace( N.log10(kmin) , N.log10(kmax) , 200 )    # spectral grid
    rr = N.logspace( N.log10(lmin) , N.log10(lmax) , 200 ) # physical grid
    kny = 2.0 * N.pi / boxsize * ng / 2.0

    # linear
    kbins,pk = N.loadtxt('data_itam/planck_pk.txt')
    pk = 10.**N.interp(N.log10(k),N.log10(kbins),N.log10(pk),left=0.,right=0.)
    Wk2 = N.exp(-k*k*Rth*Rth)
    pk *= Wk2

    xi = hankel3d.pk_to_xi( pk , rr , k , nmax , stepsize )

    print('solving integral')
    sigma2_l = xi[0]
    xi_ng = N.asarray([ solve_integral_lognormal(rrho,sigma2_l) for rrho in xi/xi[0] ])
    pkfromxi = hankel3d.xi_to_pk( xi_ng , rr , k , nmax , stepsize*10 )
    pkfromxi_th = hankel3d.xi_to_pk( N.exp(xi)-1. , rr , k , nmax , stepsize*10 )

    plt.clf()
    plt.subplot(211)
    plt.xlim([-2.,150.])
    plt.plot(rr,rr**2*(N.exp(xi)-1.),label='theoretical')
    plt.plot(rr,rr**2*xi_ng,'r--',label='numerical')
    plt.title('correlation')
    plt.legend()

    plt.subplot(212)
    plt.semilogx(k,(pkfromxi-pkfromxi_th)/pkfromxi_th,lw=2.)
    plt.xlabel(r'$k [h/{\rm Mpc}]$')
    plt.ylabel(r'$\Delta P(k)/P(k)$')
    plt.legend()
    plt.xlim([kmin,kmax/2.])
    plt.ylim([-0.01,0.01])
    plt.show()

    return 0



def test_hankel(boxsize=256., ng=256 , Rth = 2. ,  nmax=10000 , stepsize=1e-04 ):
    '''
    test the hankel transformation applied to the Nbody realization
    '''

    lmin = boxsize/float(ng)/10.
    lmax = boxsize
    kmin = 2.*N.pi/lmax
    kmax = 2.*N.pi/lmin
    print 'kny=',N.pi/boxsize*float(ng)

    k = N.logspace( N.log10(kmin) , N.log10(kmax) , 400 )    # spectral grid
    r = N.logspace( N.log10(lmin) , N.log10(lmax) , 400 )    # physical grid

    kny = 2.0*N.pi/boxsize*ng/2.0

    kbins,pk = N.loadtxt('data_itam/planck_pk.txt')
    pk = 10.**N.interp(N.log10(k),N.log10(kbins),N.log10(pk),left=0.,right=0.)
    Wk2 = N.exp(-k*k*Rth*Rth)
    pk *= Wk2

    xi = hankel3d.pk_to_xi( pk , r , k , nmax , stepsize )

    plt.clf()
    plt.subplot(211)
    plt.plot(r,r**2*xi)
    plt.xlim([-2.,50.])
    #plt.ylim([-5.,15.])
    plt.xlabel(r'$r\ [Mpc/{\rm h}]$')
    plt.ylabel(r'$r^2 \xi(r)$')

    pkfromxi = hankel3d.xi_to_pk(  xi , r , k , nmax , stepsize*10 )
    pk[k>=kny] = 0.0

    plt.subplot(212)
    plt.semilogx(k, pkfromxi/pk-1. ,label='smoothed')
    plt.legend()
    plt.xlabel(r'$k [h/{\rm Mpc}]$')
    plt.ylabel(r'$R(k)-1$')
    plt.ylim([-0.03,0.03])
    plt.show()

    return 0

def solve_integral(rho,cdf,ppf,sigma2=1):

    if rho >= 1.0:
        rho = 0.9999

    nhg = 30
    x, w = N.polynomial.hermite.hermgauss(nhg)
    Sigma = sigma2 * N.array([[1., rho], [rho, 1.]])
    Nd = 2
    const = N.pi**(-0.5*Nd)

    # gaussian variable
    xn = N.array(list(itertools.product(*(x,)*Nd)))
    
    # gauss hermite weights
    wn = N.prod(N.array(list(itertools.product(*(w,)*Nd))), 1)

    # normalized diagonal variables
    yn = 2.0**0.5*N.dot(N.linalg.cholesky(Sigma), xn.T).T

    #scipy gaussian cdf
    yn = norm.cdf( yn , scale = N.sqrt(sigma2) )
    
    # actual ppf
    gn = N.power( 10. , N.interp(yn, cdf , ppf ,left=ppf[0],right=ppf[-1]) )-1.

    gn = N.prod( gn , 1 )

    if not N.all( N.isfinite( gn ) ):
        gn[N.where(N.isinf(gn))] = 0.
        #assert 0

    corr = N.sum( (wn * const ) * gn )

    return corr


def solve_integral_lognormal(rho,sigma2=1):

    if rho >= 1.0:
        rho = 0.9999

    nhg = 30
    x, w = N.polynomial.hermite.hermgauss(nhg)
    Sigma = sigma2 * N.array([[1., rho], [rho, 1.]])
    Nd = 2
    const = N.pi**(-0.5*Nd)

    # gaussian variable
    xn = N.array(list(itertools.product(*(x,)*Nd)))

    # gauss hermite weights
    wn = N.prod(N.array(list(itertools.product(*(w,)*Nd))), 1)

    # normalized diagonal variables
    yn = 2.0**0.5*N.dot(N.linalg.cholesky(Sigma), xn.T).T

    #scipy gaussian cdf
    yn = norm.cdf( yn , scale = N.sqrt(sigma2) )

    # lognormal ppf. loc and s are the means and sigma of the gaussian that produce the lognormal 
    gn = lognorm.ppf( yn , s = N.sqrt(sigma2) ,loc=0.0, scale=1.0)

    # to have Coles and jones
    gn *= N.exp(-sigma2/2.)
    gn -= 1.

    gn = N.prod( gn , 1 )

    if not N.all( N.isfinite( gn ) ):
        gn[N.where(N.isinf(gn))] = 0.
        #assert 0

    corr = N.sum( (wn * const ) * gn )

    return corr


if __name__ == "__main__":
    test_integral()
    test_hankel(stepsize=1e-04,Rth=5.0,boxsize=500., ng=500)
    test_itam(nmax=10000,stepsize=1e-04,boxsize=256.,ng=256,Rth=1.)
    test_itam_lognormal(nmax=10000,stepsize=1e-04,boxsize=500.,ng=256,Rth=500./256.)
