from GRITS.constants import MJD_TO_DJD
import array
import numpy as np 
import ephem
from astropy.time import Time
from astropy.coordinates import get_sun,get_body

def getsunjup(file_start, file_end, ints):
    t = Time(np.linspace(file_start,file_end, ints),format='mjd')
    sunSC =  get_sun(t)
    jupSC = get_body("jupiter",t)
    sun = []
    for sc in sunSC:
        sun.append((sc.ra.deg,sc.dec.deg))
    jup = [] 
    for sc in jupSC:
        jup.append((sc.ra.deg,sc.dec.deg))
    juppos = np.array(jup)
    sunpos = np.array(sun)
    return sunpos, juppos


def getcyga(freq):
 # from Baars et al 1977
    a = 4.695 #+/-0.018
    b = 0.085 #+/-0.003
    c = -0.178 #+/-0.001
    return 10**(a+b*np.log10(freq)+c*(np.log10(freq)**2))
def getcasa(freq):
# This is not quite right, but better than Baars et al. 
# See https://lda10g.alliance.unm.edu/LWA1LowFrequencySkySurvey/
    return 1.0*getcyga(freq)

def getcrab(freq):
 # from Baars et al 1977
    a = 3.915 #+/- 0.031
    b = -0.299 # +/- 0.009
    return 10**(a+b*np.log10(freq))

def getvira(freq):
# from Kuehr et al. 1981
    a = 4.93 # +/- 0.03
    b = -0.83 # +/- 0.01
    return 10**(a + b*np.log10(freq))
