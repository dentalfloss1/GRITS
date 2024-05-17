from GRITS.OrvilleImageDB import OrvilleImageDB
from contextlib import closing
import ephem
from collections import deque
import numpy as np

from lsl.common.stations import lwasv 
from lsl.astro import MJD_OFFSET, DJD_OFFSET

rng = np.random.default_rng()
freq = [40350000., 43650000., 46950000., 50250000., 53550000., 56850000.]
ASP_CONFIG = deque([{'asp_filter': -1,
                     'asp_atten_1': -1,
                     'asp_atten_2': -1,
                     'asp_atten_s': -1},], 1)

mjd = 59430.09473379629
# Mostly taken from orville_imager.py
phase_center_ha = 1.0*ephem.hours("-0:07:59.82")
phase_center_dec = 1.0*ephem.degrees("33:21:27.5")

pce = np.sin(phase_center_dec)*np.sin(lwasv.lat) \
              + np.cos(phase_center_dec)*np.cos(lwasv.lat)*np.cos(phase_center_ha)
pce = np.arcsin(pce)
pca = np.sin(phase_center_dec) - np.sin(pce)*np.sin(lwasv.lat)
pca = pca / np.cos(pca) / np.cos(lwasv.lat)
pca = np.arccos(pca)

phase_center_az = pca
phase_center_alt = pce

def setinfo(offset=0):
    lwasv.date = mjd + (MJD_OFFSET - DJD_OFFSET) + (offset*5/60.0/60.0/24.0)
    lst = lwasv.sidereal_time()
    # mimicking the test.oims file in the Orville imager repo 
    info = {'start_time':    mjd,
                    'int_len':       5/60.0/60.0/24.0,
                    'fill':          0.0,
                    'lst':           lst * 0.5/np.pi,
                    'start_freq':    freq[0],
                    'stop_freq':     freq[-1],
                    'bandwidth':     freq[1]-freq[0],
                    'center_ra':     (lst - phase_center_ha) * 180/np.pi,
                    'center_dec':    phase_center_dec * 180/np.pi,
                    'center_az':     phase_center_az * 180/np.pi,
                    'center_alt':    phase_center_alt * 180/np.pi,
                    'pixel_size':    0.9027777777777778,
                    'stokes_params': ('I,Q,U,V')}
    info.update(ASP_CONFIG[0])
    return info

filename = "Simsource.oims"
with closing(OrvilleImageDB(filename, mode='a', station=lwasv.name)) as db:
    for off in range(0,6*5,5):
        data = np.zeros((4,6,144,144))
        data[0] =  rng.normal(loc=200,scale=10,size=(6,144,144))
        for s in range(1,4):
            data[s] = rng.normal(loc=0,scale=100,size=(6,144,144))
        data = data.transpose(1,0,2,3)
        db.add_image(setinfo(offset=off), data)
