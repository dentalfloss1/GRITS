#!/usr/bin/env python

from lsl_toolkits.OrvilleImage import OrvilleImageDB
from GRITS.constants import CYG_A, CAS_A, TAU_A, VIRGO_A
from GRITS.ateamtools import getcyga, getcasa, getcrab, getvira
from GRITS.imagetools import hdrtopixmask,pbcorr,hdrtoworld
from contextlib import closing
import ephem
from collections import deque
import numpy as np
from lsl.common.stations import lwasv 
from lsl.astro import MJD_OFFSET, DJD_OFFSET

rng = np.random.default_rng()
freq = [38700000., 42000000., 45300000., 48600000., 51900000., 55200000.]
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

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

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
if __name__=="__main__":
    filename = "stablesky.oims"
    gaussiannoise =  rng.normal(loc=200,scale=1e-3,size=(6,6,144,144))
    symgaussiannoise =  rng.normal(loc=0,scale=1e-3,size=(6,6,144,144))
    station = lwasv.name
    with closing(OrvilleImageDB(filename, mode='w', station=lwasv.name)) as sdb:
        for ind,off in enumerate(range(0,6*5,5)):
            info = setinfo(offset=off)
            data = np.zeros((4,6,144,144))
            data[0] =  gaussiannoise[ind]
            for s in range(1,4):
                data[s] =  symgaussiannoise[ind]
            data = data.transpose(1,0,2,3)
            gausscube = np.zeros((6,144,144))
            for src,name,getflux in zip([CYG_A, CAS_A, TAU_A, VIRGO_A],["cyg a","cas a","tau a","virgo a"],[getcyga, getcasa, getcrab, getvira]):
                coords = hdrtopixmask(info,data.shape[-1],360.0/info['pixel_size']/np.pi/2,src[0],src[1],station)[0]
                if not np.isnan(coords).any():
                    for c,f in enumerate(freq):
                        fwhm = 2.2*74e6/(f+3300000.0/2)/info['pixel_size']
                        thisgauss = makeGaussian(data.shape[-1],fwhm=fwhm, center=(coords[0],coords[1]))*getflux(f/1e6)
                        gausscube[c] += thisgauss
            for c in range(len(freq)):
                data[c,0,:,:] = np.where(gausscube[c] > 100, gausscube[c], data[c,0,:,:])
                XX, YY = pbcorr(info,data.shape[-1],c,info['pixel_size'],b'LWASV')
                data[c,0,:,:]*=(XX+YY)/2

            sdb.add_image(info, data*1e-3)
            
    filename = "oneofftransient.oims"
    with closing(OrvilleImageDB(filename, mode='w', station=lwasv.name)) as tdb:
        for ind,off in enumerate(range(0,6*5,5)):
            info = setinfo(offset=off)
            data = np.zeros((4,6,144,144))
            data[0] =  gaussiannoise[ind]
            for s in range(1,4):
                data[s] =  symgaussiannoise[ind]
            data = data.transpose(1,0,2,3)
            gausscube = np.zeros((6,144,144))
            for src,name,getflux in zip([CYG_A, CAS_A, TAU_A, VIRGO_A],["cyg a","cas a","tau a","virgo a"],[getcyga, getcasa, getcrab, getvira]):
                coords = hdrtopixmask(info,data.shape[-1],360.0/info['pixel_size']/np.pi/2,src[0],src[1],station)[0]
                if not np.isnan(coords).any():
                    for c,f in enumerate(freq):
                        fwhm = 2.2*74e6/(f+3300000.0/2)/info['pixel_size']
                        thisgauss = makeGaussian(data.shape[-1],fwhm=fwhm, center=(coords[0],coords[1]))*getflux(f/1e6)
                        gausscube[c] += thisgauss
            if ind==3:
                for c,f in enumerate(freq):
                    fwhm = 2.2*74e6/(f+3300000.0/2)/info['pixel_size']
                    thisgauss = makeGaussian(data.shape[-1],fwhm=fwhm, center=(75,65))*2000
                    gausscube[c] += thisgauss
            for c in range(len(freq)):
                data[c,0,:,:] = np.where(gausscube[c] > 100, gausscube[c], data[c,0,:,:])
            for c in range(len(freq)):
                XX, YY = pbcorr(info,data.shape[-1],c,info['pixel_size'],b'LWASV')
                data[c,0,:,:]= data[c,0,:,:]*(XX+YY)/2
            data[0]= 0
            tdb.add_image(info, data*1e-3)
        
            
