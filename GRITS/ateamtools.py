from GRITS.constants import MJD_TO_DJD
import array
import numpy as np 
import ephem

def getsunjup(file_start, file_end, ints):
    djd_start = file_start + MJD_TO_DJD
    duration = int(round((file_end - file_start)*24*3600)) # in seconds
    sunra = array.array('f')
    sundec = array.array('f')
    jupra = array.array('f')
    jupdec = array.array('f')
    for i,t in enumerate(range(0,duration*(ints),duration)):
        date = djd_start + (t/24/3600) - (duration/2/24/3600) # in days 
        sun = ephem.Sun(date)
        sunra.append(sun.ra*180/np.pi)
        sundec.append(sun.dec*180/np.pi)
        jup = ephem.Jupiter(date)
        jupra.append(jup.ra*180/np.pi)
        jupdec.append(jup.dec*180/np.pi)
    juppos = np.array([(ra,dec) for ra,dec in zip(jupra,jupdec)])
    sunpos = np.array([(ra,dec) for ra,dec in zip(sunra,sundec)])
    return sunpos, juppos



def getcasa(freq):
 # from Baars et al 1977
    a = 5.625 #+/-0.021
    b = -0.634 #+/-0.015
    c = -0.023 #+/-0.001
    return 10**(a+b*np.log10(freq)+c*(np.log10(freq)**2))
def getcyga(freq):
 # from Baars et al 1977
    a = 4.695 #+/-0.018
    b = 0.085 #+/-0.003
    c = -0.178 #+/-0.001
    return 10**(a+b*np.log10(freq)+c*(np.log10(freq)**2))
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
