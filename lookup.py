'''
   These functions are not necessary to run itam.py, but they are used to write the txt of 
   the lookup tables I used. Essentially one needs a table for 
   - the power spectrum
   - the percentile point function
   - rescaling factor 
  
  In *lookup_Pk* one needs to specify the cosmological parameters of the chosen simulation.
  In *lookup_ppf* and *rescale_factor* one needs to specify a path to a file containing the density field,
  and the smoothing scale one wants to use (it has to be the same in itam.py as well).
'''

import numpy as N
from classy import Class
from scipy.stats import rankdata
from scipy.integrate import quad

def lookup_Pk(cosmology='planck',nonlinear=0):
    """
    it saves the lookup table of the (non) linear power spectrum generate from CLASS.
    If nonlinear is False (default) it generates the linear power spectrum.
    You can choose between
    - planck
    - wmap
    - ML
    Choose also whether you want a nonlinear power spectrum, default is linear (nonlinear=0)
    """

    # k in h/Mpc
    k = N.logspace(-4., 3., 3*1024)

    if nonlinear==1:
        hf = 'halofit'
        saveto = 'data_itam/'+cosmology+'_pk.txt'

    else:
        hf = ''
        saveto = 'data_itam/'+cosmology+'_pk_linear.txt'

    if cosmology == 'planck':
        class_params = {
        'non linear': hf,
        'output': ['mPk','vTk'],
        'P_k_max_1/Mpc': 1000.,
        'z_pk': 0.,
        'A_s': 2.3e-9,
        'n_s': 0.96,
        'h': 0.7,
        'omega_b': 0.0225,
        'Omega_cdm': 0.25,
        }
        sig8_0 = 0.8


    elif cosmology == 'wmap':
        class_params = {
        'non linear': hf,
        'output': ['mPk','vTk'],
        'P_k_max_1/Mpc': 1000.,
        'z_pk': 0.,
        'A_s': 2.3e-9,
        'n_s': 0.967,
        'h': 0.704,
        'omega_b': 0.02253,
        'Omega_cdm': 0.226,
        }
        sig8_0 = 0.81


    elif cosmology == 'ML':
        class_params = {
        'non linear': hf,
        'output': ['mPk','vTk'],
        'P_k_max_1/Mpc': 1000.,
        'z_pk': 0.,
        'A_s': 2.3e-9,
        'n_s': 1.,
        'h': 0.73,
        'omega_b': 0.045*0.73**2,
        'Omega_cdm': 0.25-0.045,
        }
        sig8_0 = 0.9

    else:
        raise ValueError("the cosmology you chose does not exist")

    cosmoClass_nl = Class()
    cosmoClass_nl.set(class_params)
    cosmoClass_nl.compute()

    # rescale the normalization of matter power spectrum to have sig8=0.8 today
    sig8 = cosmoClass_nl.sigma8()
    A_s = cosmoClass_nl.pars['A_s']
    cosmoClass_nl.struct_cleanup() # does not clean the input class_params, cosmo.empty() does that
    cosmoClass_nl.set(A_s=A_s*(sig8_0*1./sig8)**2)
    cosmoClass_nl.compute()

    h = cosmoClass_nl.pars['h']
    pk_nl = N.asarray([ cosmoClass_nl.pk(x*h, 0.,)*h**3 for x in k ])

    kpk = N.vstack((k,pk_nl))
        
    N.savetxt(saveto,kpk)
    print('saving', saveto )
    return


def lookup_ppf(nsamples=1e+05, boxsize=256.0, Rth=1.,density=None, pathpk='data_itam/planck_pk.txt',  saveto_ppf='data_itam/_ppf.txt',
        saveto_rescale='data_itam/rescale_factor.txt'):
    '''
    Inputs::
      nsamples: the number of points at which ppf is sampled\.
      density: path to a binary with the unsmoothed density field.
      Rth: smoothing scale of the Gaussian kernel.
    Outputs::
      - Lookup table for the point percent function of density field of simulation
      - rescale factor to make variance of PDF and Pk consistent
    '''
    try:
        d = N.load(density) # shape is (ng,ng,ng)
    except:
        ValueError("the simulation you chose does not exist")

    dnlR = smoothfield(d,boxsize,Rth).flatten()
    cdvars = N.var(dnlR,axis=None)
    dnlR = N.log10(1.+dnlR)
    dnlR = dnlR[N.isfinite(dnlR)]
    rk = rankdata(dnlR)
    rk /= len(rk)+1.
    rksort = N.sort(rk)
    dsort = N.sort(dnlR)
    cdf = N.linspace(rksort[0],rksort[-1],nsamples)
    ppf = N.interp(cdf,rksort,dsort,left=dsort[0],right=dsort[-1])
    #plt.plot(ppf,cdf);plt.show();
    ppf = N.vstack((cdf,ppf))
    print('saving', saveto_ppf)
    N.savetxt(saveto_ppf,ppf)

    kbins,Pk = N.loadtxt(pathpk)

    def integrand(k, R=Rth):
        pk = 10**N.interp(N.log10(k),N.log10(kbins),N.log10(Pk),left=0.,right=0.)
        ret = pk * N.exp(-(k * R )**2)* k**2.
        return ret
    
    ng = N.shape(d)[0]
    k_min = 2.0 * N.pi / boxsize
    k_max = 2.0 * N.pi / boxsize * ng / 2.

    args = Rth
    sigma2, _ = quad(integrand, k_min, k_max, args = args, epsabs = 1e-08, epsrel = 1e-08, limit = 100)
    sigma2 /= 2.0 * N.pi**2.
    print( 'cell density variance=', cdvars , 'theoretical expected variance', sigma2 )
    ratio = cdvars / sigma2
    print('ratio=', ratio )
    print('saving', saveto_rescale)
    N.savetxt( saveto_rescale , [ratio] )

    return


def smoothfield( d , boxsize, Rth ):
    """ smooth a field on a grid, with smoothing scale Rth, using a guassian kernel """

    ng = N.shape(d)[0]
    dk  = N.fft.rfftn(d)
    kgrid = getkgrid(boxsize,ng)
    dk = dk * N.exp( -(kgrid * Rth) ** 2. / 2. )
    d = N.fft.irfftn(dk)

    return d


def getkgrid(boxsize,ng):
    '''
    It returns a grid of modulus k in fft format
    '''
    boxsize = float(boxsize)
    ng = int(ng)
    kmin = 2*N.pi/N.float(boxsize)
    thirdim = ng/2 + 1

    sh = (ng,ng,thirdim,)
    kx,ky,kz = N.mgrid[0:sh[0],0:sh[1],0:sh[2]].astype(float)

    kx[N.where(kx > ng/2)] -= ng
    ky[N.where(ky > ng/2)] -= ng
    kz[N.where(kz > ng/2)] -= ng

    kx *= kmin
    ky *= kmin
    kz *= kmin

    k = N.sqrt(kx**2+ky**2+kz**2)

    return k

if __name__ == "__main__":
    lookup_Pk(cosmology='planck',nonlinear=1)
    #lookup_ppf(nsamples=1e+05, boxsize=256.0, Rth=1.0,density='data_itam/density.npy', pathpk='data_itam/planck_pk.txt',  saveto_ppf='data_itam/_ppf.txt',
    #    saveto_rescale='data_itam/rescale_factor.txt')
