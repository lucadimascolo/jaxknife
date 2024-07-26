import matplotlib.pyplot as plt

import numpy as np; rng = np.random.default_rng(seed=678)

from casatasks import simobserve
from casatools import ms, componentlist
import jaxknife
import os

import scipy.stats

srcname = 'test'
conf, tottime = 3, '1h'

ms = ms()
cl = componentlist()

if not os.path.exists(f'output/{srcname}_{tottime}_C{conf}'):
    ptgname = 'J2000 00h00m00.000s -00d00m00.00s'

    ptgfile = open(f'{srcname}.ptg','w')
    ptgfile.write(f'{ptgname}\n')
    ptgfile.close()

    cl.done()
    cl.addcomponent(dir=ptgname,flux=0.10,fluxunit='Jy',freq='100.0GHz',shape='point')
    cl.rename(f'{srcname}.cl')
    cl.done()

    cfglist = f'jaxknife/config/C11/alma.cycle11.{conf}.cfg'

    simobserve(project = f'{srcname}_{tottime}_C{conf}',
             compwidth = '1.875GHz',
              complist = f'{srcname}.cl',
          setpointings = False,
               ptgfile = f'{srcname}.ptg',
             overwrite = True,
           integration = '10s',
             totaltime = tottime,
             direction = ptgname,
              incenter = '100GHz',
               inwidth = '1.875GHz',
              inbright = '',
           antennalist = cfglist,
               obsmode = 'int',
                  seed = rng.integers(100,10000),
              graphics = 'none')

    os.system('rm -rf *.ptg *.cl *.last *log')
    os.system('mkdir -p output')
    os.system(f'mv {srcname}_{tottime}_C{conf} output/')

visname = f'output/{srcname}_{tottime}_C{conf}/{srcname}_{tottime}_C{conf}.alma.cycle11.{conf}'

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

mr = scipy.stats.median_abs_deviation(jk.image(),axis=None,scale='normal')

mj = np.zeros(1000)
for mi in range(mj.shape[0]):
    mj[mi] = scipy.stats.median_abs_deviation(jk.run(),axis=None,scale='normal')

plt.hist(mj,bins=40)
plt.axvline(mr,color='r')
plt.show(); plt.close()
          
os.system('rm -vf *.log')