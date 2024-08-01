from casatools import ms; ms = ms()

from astropy import constants as const
from astropy import units as u

import jax; jax.config.update('jax_enable_x64',True)
import jax.numpy as jp

import jax_finufft

import numpy as np
import scipy.fft

class build:
    def __init__(self,vis,spw=0,field=0,cdelt=None,csize=None,seed=None,):
        self.rng = np.random.default_rng(seed=seed)
        
        print('Importing uv data...')

        ms.open(vis,nomodify=True)
        ms.selectinit(int(spw))
        ms.select({'field_id' : int(field)})

        freqs = ms.range('chan_freq')['chan_freq'][:,0]
        rec = ms.getdata(['u','v','data'])
        
        self.u = np.copy(rec['u'])*freqs[0]/const.c.to('m/s').value
        self.v = np.copy(rec['v'])*freqs[0]/const.c.to('m/s').value

        self.data = np.copy(rec['data'])
        self.data = np.mean(self.data[:,0,:],axis=0)
        ms.close()

        if cdelt is None:
            uvdist = np.hypot(self.u,self.v)

            cdelt = 3.60E+03*np.rad2deg(0.25/np.nanmax(uvdist))
            cdelt = int(np.round(cdelt,2)/5.00E-05)*5.00E-05

            cdelt = cdelt*u.arcsec
        
        if csize is None:
            csize = 2.44*const.c.to('m/s').value/np.nanmin(freqs)/12.00
            csize = 3.60E+03*np.rad2deg(1.50*csize)
            csize = csize*u.arcsec
            csize = scipy.fft.next_fast_len(int((csize/cdelt).to(u.dimensionless_unscaled).value))
        self.csize = csize

        del uvdist

        self.cdelt = cdelt
        self.csize = csize

        self.x = -2.00*np.pi*self.v*np.deg2rad(cdelt.to('deg').value)
        self.y =  2.00*np.pi*self.u*np.deg2rad(cdelt.to('deg').value)

        self.data = jp.array(self.data)
        self.x = jp.array(self.x)
        self.y = jp.array(self.y)

    def run(self):
        flips = np.ones(self.data.shape[0])
        flips[:self.data.shape[0]//2] = -1.00
        self.rng.shuffle(flips)

        return self.image(self.data*flips)
    
    def image(self,c=None):
        if c is None: c = self.data
        x = jp.append(self.x,-self.x)
        y = jp.append(self.y,-self.y)
        c = jp.append(c,c.conj())
        return jax_finufft.nufft1((self.csize,self.csize),c/np.size(c),x,y).real