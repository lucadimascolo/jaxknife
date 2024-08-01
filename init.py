import matplotlib.pyplot as plt

import numpy as np; rng = np.random.default_rng(seed=678)

from casatasks import simobserve
from casatools import ms, componentlist
import jaxknife
import os

from astropy.utils.console import ProgressBar

import scipy.stats

ms = ms()
cl = componentlist()

import cmocean

cmap = plt.get_cmap('cmo.deep_r')

conf, tottime = 4, '1h'

nsobs = 200
njack = 1000

with ProgressBar(nsobs*njack) as bar:
    for iter in range(nsobs):
        seed = rng.integers(100,10000)

        srcname = 'test_{0:05d}'.format(seed)

        if not os.path.exists(f'output/{srcname}_{tottime}_C{conf}'):
            ptgname = 'J2000 00h00m00.000s -00d00m00.00s'

            ptgfile = open(f'{srcname}.ptg','w')
            ptgfile.write(f'{ptgname}\n')
            ptgfile.close()

            cl.done()
            cl.addcomponent(dir=ptgname,flux=0.10,fluxunit='Jy',freq='279.0GHz',shape='point')
            cl.rename(f'{srcname}.cl')
            cl.done()

            cfglist = f'jaxknife/config/C07/alma.cycle7.{conf}.cfg'

            simobserve(project = f'{srcname}_{tottime}_C{conf}',
                    compwidth = '31MHz',
                    complist = f'{srcname}.cl',
                setpointings = False,
                    ptgfile = f'{srcname}.ptg',
                    overwrite = True,
                integration = '10s',
                    totaltime = tottime,
                    direction = ptgname,
                    incenter = '279GHz',
                    inwidth = '31MHz',
                    inbright = '',
                antennalist = cfglist,
                    obsmode = 'int',
                        seed = seed,
                    graphics = 'none')

            # scale by sqrt(6)

            os.system('rm -rf *.ptg *.cl *.last *log')
            os.system('mkdir -p output')
            os.system(f'mv {srcname}_{tottime}_C{conf} output/')

        visname = f'output/{srcname}_{tottime}_C{conf}/{srcname}_{tottime}_C{conf}.alma.cycle7.{conf}'

        if not os.path.exists(f'{visname}.noise.ms'):
            os.system(f'cp -r {visname}.noisy.ms \
                            {visname}.noise.ms')

            ms.open(f'{visname}.ms')
            rec = ms.getdata(['data'])
            data_clean = np.copy(rec['data'])
            ms.close(); del rec

            ms.open(f'{visname}.noise.ms',nomodify=False)
            rec = ms.getdata(['data'])
            rec['data'] = rec['data']-data_clean
            ms.putdata(rec)
            ms.close(); del rec

        jk = jaxknife.build(f'{visname}.noise.ms')

        getstd = lambda x: scipy.stats.median_abs_deviation(x,axis=None,scale='normal')
        getstd = lambda x: np.std(x)

        mr = jk.image()
        mj = jk.run()

        if False:
            plt.subplot(121); plt.imshow(mr,origin='lower',cmap='cmo.balance',vmin=-np.nanmax(np.abs(mr)),vmax=np.nanmax(np.abs(mr)))
            plt.subplot(122); plt.imshow(mj,origin='lower',cmap='cmo.balance',vmin=-np.nanmax(np.abs(mr)),vmax=np.nanmax(np.abs(mr)))
            plt.show(); plt.close() 
        else:
            mr = getstd(jk.image())

        #   print('simobserve RMS > {0:8.2E}'.format(mr))
            
            if not os.path.exists(f'figures/test_{seed:05d}.npy'):
                mj = np.zeros(njack)
                for mi in range(mj.shape[0]):
                    bar.update()
                    mj[mi] = getstd(jk.run())
        #           print('jackknife RMS  > {0:8.2E} [{1:04d}/{2:04d}]'.format(mj[mi],mi+1,mj.shape[0]),end='\r')

                np.save(f'figures/test_{seed:05d}.npy',mj)
                print('')
            else:
                mj = np.load(f'figures/test_{seed:05d}.npy')
                for mi in range(mj.shape[0]): bar.update()

            plt.hist(mj,bins=40,color=cmap(iter/20),density=True)
            plt.axvline(mr,color='black',lw=0.50)

# plt.xlim(1.62E-05-0.09E-05,
#          1.62E-05+0.09E-05)

plt.xlabel('Standard deviation [Jy]')
plt.savefig('figures/test_total.pdf',format='pdf',dpi=300); plt.close()

os.system('rm -vf *.log')