#!/usr/bin/env python
from tqdm import tqdm
import math
import os
import sys
import datetime
import traceback
start_epoch = datetime.datetime(1858, 11, 17, 00, 00, 00, 00)
import argparse
import sys
from configparser import ConfigParser 
from contextlib import closing
from bifrost.device import set_device

import matplotlib as mpl
font = {'weight' : 'bold',
        'size'   : 22}
mpl.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from lsl.common.mcs import mjdmpm_to_datetime
from lsl.misc import parser as aph
from lsl import astro

from scipy.stats import norm
from scipy.interpolate import interp1d

import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time 

from bifrost import ndarray as BFArray
from bifrost import map as BFMap 
from bifrost.device import set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

from scipy.optimize import curve_fit

from GRITS.ateamtools import getsunjup, getcasa, getcyga, getcrab, getvira
from GRITS.catalogs import cat3c, VLSSr50jy
from GRITS.configtemplate import configfile
from GRITS.constants import floatmax
from GRITS.GPUimagestats import getminrms,fluxcalmask,getmultistat,getdiffstats
from GRITS.GPUgrouping import groupvertcols,grouphorizrows,groupdiagcols,groupdiagrows,labelgroups
from GRITS.imagetools import  angsep,maskhorizon, pbcorr, hdrtopixmask, hdrtoworld, maskislands
from GRITS.plotresults import plotresults, plotLC
from GRITS.qc import restrictnumsrc,restrictangvel, checkgalactic
from GRITS.readimage import readoims,readpims
from GRITS.fitGaussian import fitGaussian

from lsl_toolkits.OrvilleImage import OrvilleImageDB
from lsl_toolkits.PasiImage import PasiImageDB
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("error",category=RuntimeWarning)

def gaussian_nobackgound(xy, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()
def gaussian_backgound(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()


def main(src,config,quiet=False,skip=0,endearly=None,maskstokes=None,positioncheck=False):
    filename = src['filename']
    uniquesrcid=0
    srclist = []
    obslist = []
    nstokes = 4 
    chunking = int( config['Batch Control']['chunktime'])
    if not quiet:
        print(filename)

            
    if ".oims" in filename:
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readoims(filename)
    elif ".pims" in filename:
        ints,file_start,file_end,ngrid,psize,nchan,sRad,hdrlist,data,imagedata,station = readpims(filename)
    else:
        print("unrecognized filename")
        exit()
        
    if ints==0:
        raise Exception("file is empty")
    for c in range(nchan):
        XX,YY = pbcorr(hdrlist[0],ngrid,c,psize,station)
        imagedata[:,c,0,:,:]/=((XX+YY)/2)
        data[:,c,0,:,:]/=((XX+YY)/2)

        
    if endearly is None:
        endearly = ints+1
    t1 = datetime.datetime.now()
    # Get sun and jupiter position for every time in the file. 
    # This part of the code could use some optimization. 
    # There is FORTRAN code that is publically available to
    # compute these positions. If it seems necessary,
    # I could attempt to adapt that code to the GPU
    sunpos, juppos = getsunjup(file_start, file_end, ints)
    t2 = datetime.datetime.now()
    if not quiet:
        print("Get sun/jup:",t2-t1)

    t1 = datetime.datetime.now()


    t2 = datetime.datetime.now()
    if not quiet:
        print("Mask Horizon:",t2-t1)
    
    t1 = datetime.datetime.now()
    minrms = getminrms(sRad, imagedata,ints,nchan,ngrid,station, chunking=chunking)
    t2 = datetime.datetime.now()
    if not quiet:
        print("Compute RMS:",t2-t1)
    t1 = datetime.datetime.now()
    # Flux calibrate data and mask off A-Team
    imagedata, data, corrfluxlist = fluxcalmask(sRad, data, imagedata, hdrlist, minrms[:,:,0], sunpos, juppos, nchan, ngrid, station, filename, positioncheck=positioncheck)
    t2 = datetime.datetime.now()
    if not quiet:
        print("calibrate:",t2-t1)
    # UN pbcorrect diff ims for a flat sky. We measure fluxes using the image data, so we keep that as pb corrected
    t1 = datetime.datetime.now()
    if not quiet:
        print("corrflux:",corrfluxlist)
    removechan = []
    outchannels=[]
    counter = 0
    flaggedtimes = [] 
    for i,c in enumerate(corrfluxlist):
        if c<=0:
            removechan.append(c)
            for i in range(len(imagedata)):
                flaggedtimes.append((i,c))
        else:
            outchannels.append(i)
         #    rmsdata[:,counter] = minrms[:,i]*c
            counter+=1

    t2 = datetime.datetime.now()
    if not quiet:
        print("un-pbcorr and remove bad channels:",t2-t1)
    nchan = imagedata.shape[1]
    t1 = datetime.datetime.now()
    rmsdata = getminrms(sRad, data,ints,nchan,ngrid,station, chunking=chunking)
    t2 = datetime.datetime.now()
    if not quiet:
        print("RE-compute RMS:",t2-t1)
    if not quiet:
        try:
            medrms = np.median(rmsdata[1:])
            maxrms = np.amax(rmsdata[1:])
            print("min","max","median")
            print(np.amin(rmsdata[1:]),maxrms,medrms)
        except RuntimeWarning as e:
            print(e)
            print('could not compute median and max rms')
    for c in range(nchan):
        for s in range(4):
            flaggedtimes.append((0,c,s))
    for ind in range(ints): 
        if ((ind) < skip):
            for c in range(nchan):
                for s in range(4):
                    flaggedtimes.append((ind,c,s))
        if ((ind) > endearly):
            for c in range(nchan):
                for s in range(4):
                    flaggedtimes.append((ind,c,s))
    maxDR=float(config['Quality Control']['DynamicRange'])
    for i in range(1,ints):
        for j in range(nchan):
            for k in range(nstokes):
                nomaskimdiff = imagedata[i,j,k,:,:] - imagedata[i-1,j,k,:,:]
                if rmsdata[i,j,k] == 0:
                    flaggedtimes.append((i,j,k))
                else:
                    # Take absolute value since extreme negative values can also cause problems. 
                    s2n = np.amax(np.abs(nomaskimdiff))/rmsdata[i,j,k]
                    maxyplane, maxxplane = np.where(np.abs(nomaskimdiff)==np.amax(np.abs(nomaskimdiff)))
                    if angsep(hdrlist[i]['center_ra'],hdrlist[i]['center_dec'],src['ra'],src['dec']) > (90*float(config['Quality Control']['pbpercent'])):
                        flaggedtimes.append((i,j,k))
                        continue
                    targetx, targety = hdrtopixmask(hdrlist[i],ngrid,sRad,src['ra'],src['dec'],station)[0]
                    srcsep = np.sqrt( (targetx-maxxplane)**2 + (targety-maxyplane)**2)

                    if (s2n > maxDR) and (srcsep > float(config['Quality Control']['statsep'])):
                        flaggedtimes.append((i,j,k))
    imagefit = []
    diffFIT = []
    # Source fit the images
    xarr = np.linspace(0,data.shape[-1]-1,data.shape[-1])
    yarr = np.linspace(0, data.shape[-1]-1,data.shape[-1])
    x, y = np.meshgrid(xarr, yarr)
    data[1:] = np.where(np.isnan(data[1:]),imagedata[1:] - imagedata[:-1],data[1:])
    print(filename,src['index'])
    for i in tqdm(range(ints)):
        targetx, targety = hdrtopixmask(hdrlist[i],ngrid,sRad,src['ra'],src['dec'],station)[0]
        xloc = x[np.sqrt((targetx-x)**2 + (targety-y)**2) < float(config['Quality Control']['statsep'])].astype(int)
        yloc = y[np.sqrt((targetx-x)**2 + (targety-y)**2) < float(config['Quality Control']['statsep'])].astype(int)
        for xpix,ypix in zip(xloc,yloc):
            pixflux = data[i,j,k,ypix,xpix]
            if pixflux==np.amax(data[i,j,k,np.round(yloc).astype(int),np.round(xloc).astype(int)]):
                avx = xpix
                avy = ypix
        for j in range(nchan):
            for k in range(nstokes):
                if (i,j,k) in flaggedtimes:
                    continue
                hdr = hdrlist[i]
                try:
                    midfreq = (hdr['freq']  + j*hdr['bandwidth'] + (hdr['bandwidth']/2))
                except KeyError:
                    midfreq = (hdr['start_freq']  + j*hdr['bandwidth'] + (hdr['bandwidth']/2))
                bsizepix = (2.2*74e6/midfreq/2)/hdr['psize']
                # Fit image data
                f0 = imagedata[i,j,k,int(round(avy)),int(round(avx))]
                x0 = avx
                y0 = avy
                initial_guess = [f0,x0,y0,bsizepix,bsizepix,np.pi,0.1*f0]
                if k==0:
                    if f0<0:
                        continue
                    initbounds = [(f0*0.1,f0*10),(x0-1.5,x0+1.5),(y0-1.5,y0+1.5),(1.5,6),(1.5,6),(0,2*np.pi),(0,f0*0.5)]
                else:
                    initbounds = [(-np.abs(f0)*10,np.abs(f0)*10),(x0-1.5,x0+1.5),(y0-1.5,y0+1.5),(1.5,6),(1.5,6),(0,2*np.pi),(-np.abs(f0)*0.5,np.abs(f0)*0.5)]
                bounds0 = tuple([b[0] for b in initbounds])
                bounds1 = tuple([b[1] for b in initbounds])
                bounds = [bounds0,bounds1]
                fitdata = np.transpose(imagedata[i,j,k]).ravel()
                try:
                    popt, pcov = curve_fit(gaussian_backgound, (x, y), fitdata, p0=initial_guess,bounds=bounds)
                    raOUT, decOUT = hdrtoworld(hdrlist[i],ngrid,sRad,popt[1],popt[2],station)[0]
                    param_name = ["f0","x0","y0","sigma_x","sigma_y","theta","offset"]
                    row = []
                    for pind in range(len(popt)):
                        pname = param_name[pind]
                        param = popt[pind]
                        perr = np.absolute(pcov[pind][pind])**0.5
                        row.append(pname)
                        row.append(param)
                        row.append(perr)
                    row.append(hdrlist[i]['start_time'])
                    row.append(midfreq/1e6)
                    row.append(k)
                    row.append(raOUT)
                    row.append(decOUT)
                    imagefit.append(tuple(row))
                except Exception as e:
                    print(e)

                # fit diff data
                f0 = data[i,j,k,int(round(avy)),int(round(avx))]
                x0 = avx
                y0 = avy
                initial_guess = [f0,x0,y0,bsizepix,bsizepix,np.pi]
                if k==0:
                    if f0<0:
                        continue
                    initbounds = [(f0*0.1,f0*10),(x0-1.5,x0+1.5),(y0-1.5,y0+1.5),(1.5,6),(1.5,6),(0,2*np.pi)]
                else:
                    initbounds = [(-np.abs(f0)*10,np.abs(f0)*10),(x0-1.5,x0+1.5),(y0-1.5,y0+1.5),(1.5,6),(1.5,6),(0,2*np.pi)]
                bounds0 = tuple([b[0] for b in initbounds])
                bounds1 = tuple([b[1] for b in initbounds])
                bounds = [bounds0,bounds1]
                fitdata = np.transpose(data[i,j,k]).ravel()
                try:
                    popt, pcov = curve_fit(gaussian_nobackgound, (x, y), fitdata, p0=initial_guess,bounds=bounds)
                    raOUT, decOUT = hdrtoworld(hdrlist[i],ngrid,sRad,popt[1],popt[2],station)[0]
                    param_name = ["f0","x0","y0","sigma_x","sigma_y","theta"]
                    row = []
                    for pind in range(len(popt)):
                        pname = param_name[pind]
                        param = popt[pind]
                        perr = np.absolute(pcov[pind][pind])**0.5
                        row.append(pname)
                        row.append(param)
                        row.append(perr)
                    row.append(hdrlist[i]['start_time'])
                    row.append(midfreq/1e6)
                    row.append(k)
                    row.append(raOUT)
                    row.append(decOUT)
                    diffFIT.append(tuple(row))
                except Exception as e:
                    print(e)
                    
    print(imagefit)
    print(diffFIT)


             

    return rmsdata, imagefit, diffFIT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='convert the images contained in one or more .oims files into FITS images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--configfile', type=str,
                        default='config.ini',
                        help="Configuration file.")
    parser.add_argument('--device', type=int, default=0,
                        help="Use this GPU device number")
    parser.add_argument('-q','--quiet', action='store_true')
    parser.add_argument('--skip',type=int,default=0)
    parser.add_argument('--endearly',type=int,default=None)
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='be verbose during the conversion')
    parser.add_argument("--sourcefile", type=str, required=True,
                        help="File containing targets to measure in format: index,ra,dec,stokes,filename\n")
    args = parser.parse_args()
    config = ConfigParser()
    set_device(args.device)
    if not os.path.isfile(args.configfile):
        print(f"{args.configfile} does not exist. Creating one and stopping.")
        with open(args.configfile,"w") as f:
            f.write(configfile)
    else:
        config.read(args.configfile)
        sources = np.loadtxt(args.sourcefile, delimiter=',', dtype=[('index','i4'),('ra','f8'),('dec','f8'),('stokesno','i4'),('filename','<U256')])
        for srcno, src in enumerate(sources):
            rmsdata, imagefit, diffFIT = main(src,config,quiet=args.quiet,skip=args.skip,endearly=args.endearly)
            np.save(f"src_row{srcno}_rmsdata.npy",rmsdata)
            with open(f"src_row{srcno}_imageLC.csv","w") as file:
                for row in imagefit:
                    rowout = str(row[0])
                    for col in row[1:]:
                        rowout = rowout + ","+ str(col)
                    file.write(rowout+"\n")
            with open(f"src_row{srcno}_diffLC.csv","w") as file:
                for row in diffFIT:
                    rowout = str(row[0])
                    for col in row[1:]:
                        rowout = rowout + ","+ str(col)
                    file.write(rowout+"\n")


